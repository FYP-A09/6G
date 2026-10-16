"""Single-agent vs multi-agent learners for the unified slicing environment.

SingleAgentDDPG: one controller. State = all N cells' observations concatenated,
    action = all N cells' allocations. Input and output size grow with N.
SharedMADDPG: one actor shared by every cell, acting on that cell's own
    observation (decentralised execution). The critic is centralised over the
    neighbourhood: it sees the cell's observation and action plus the mean
    observation and action of its grid neighbours (MADDPG-style centralised
    training, in mean-field form so it scales with N). Trained on the team
    reward (own reward mixed with neighbours').

Both use the same DDPG machinery (target networks, replay, logit-space noise),
the same observations (including the forecast when provided) and the same number
of gradient updates, so the comparison isolates the controller structure.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

A = 4  # eMBB, URLLC, mMTC, idle


def mlp(inp: int, out: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inp, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out))


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for t, s in zip(target.parameters(), source.parameters()):
            t.mul_(1 - tau).add_(s, alpha=tau)


@dataclass
class TrainConfig:
    episodes: int = 300
    gamma: float = 0.9
    tau: float = 0.01
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    noise_start: float = 1.0
    noise_end: float = 0.1
    warmup_steps: int = 500
    buffer_steps: int = 50_000
    seed: int = 0


class _Replay:
    def __init__(self, capacity: int, n: int, f: int):
        self.o = np.zeros((capacity, n, f), np.float32)
        self.a = np.zeros((capacity, n, A), np.float32)
        self.r = np.zeros((capacity, n), np.float32)   # team reward per cell
        self.rs = np.zeros(capacity, np.float32)        # system reward (mean over cells)
        self.o2 = np.zeros((capacity, n, f), np.float32)
        self.d = np.zeros(capacity, np.float32)
        self.i, self.full, self.cap = 0, False, capacity

    def add(self, o, a, r, rs, o2, d):
        k = self.i
        self.o[k], self.a[k], self.r[k], self.rs[k], self.o2[k], self.d[k] = o, a, r, rs, o2, d
        self.i = (k + 1) % self.cap
        self.full = self.full or self.i == 0

    def sample(self, b: int, rng: np.random.Generator):
        idx = rng.integers(0, self.cap if self.full else self.i, size=b)
        return [torch.from_numpy(x[idx]) for x in (self.o, self.a, self.r, self.rs, self.o2, self.d)]


class _Base:
    batch_steps: int

    def __init__(self, n: int, f: int, nbr: np.ndarray, cfg: TrainConfig):
        self.n, self.f, self.cfg = n, f, cfg
        self.M = torch.from_numpy(nbr.astype(np.float32))
        self.rng = np.random.default_rng(cfg.seed)
        torch.manual_seed(cfg.seed)
        self.replay = _Replay(cfg.buffer_steps, n, f)

    def _noisy(self, logits: torch.Tensor, sigma: float) -> np.ndarray:
        if sigma > 0:
            logits = logits + sigma * torch.randn_like(logits)
        return torch.softmax(logits, dim=-1).numpy()

    def train(self, env, log_every: int = 50) -> dict:
        cfg, steps, t0 = self.cfg, 0, time.time()
        returns = []
        for ep in range(cfg.episodes):
            sigma = cfg.noise_start + (cfg.noise_end - cfg.noise_start) * ep / max(cfg.episodes - 1, 1)
            obs, done, ep_r = env.reset(), False, []
            while not done:
                act = self.act(obs, sigma)
                obs2, r, team, done, _ = env.step(act)
                self.replay.add(obs, act, team, r.mean(), obs2, float(done))
                obs = obs2
                ep_r.append(r.mean())
                steps += 1
                if steps > cfg.warmup_steps:
                    self.update()
            returns.append(float(np.mean(ep_r)))
            if (ep + 1) % log_every == 0:
                print(f"  {type(self).__name__} N={self.n} seed={cfg.seed} ep {ep + 1}: "
                      f"mean reward (last {log_every}) {np.mean(returns[-log_every:]):.4f}", flush=True)
        return {"episode_reward": returns, "train_seconds": time.time() - t0, "updates": max(0, steps - cfg.warmup_steps)}

    def policy(self, env, obs):
        return self.act(obs, 0.0)


class SingleAgentDDPG(_Base):
    batch_steps = 128

    def __init__(self, n, f, nbr, cfg=TrainConfig(), hidden=256):
        super().__init__(n, f, nbr, cfg)
        self.actor, self.actor_t = mlp(n * f, n * A, hidden), mlp(n * f, n * A, hidden)
        self.critic, self.critic_t = mlp(n * (f + A), 1, hidden), mlp(n * (f + A), 1, hidden)
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.critic_t.load_state_dict(self.critic.state_dict())
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

    def _pi(self, net, o):  # o [B, N, F] -> [B, N, A]
        return torch.softmax(net(o.reshape(o.shape[0], -1)).reshape(-1, self.n, A), dim=-1)

    def _q(self, net, o, a):
        return net(torch.cat([o.reshape(o.shape[0], -1), a.reshape(a.shape[0], -1)], dim=1)).squeeze(-1)

    def act(self, obs, sigma):
        with torch.no_grad():
            logits = self.actor(torch.from_numpy(obs).reshape(1, -1)).reshape(self.n, A)
        return self._noisy(logits, sigma)

    def update(self):
        o, a, _, rs, o2, d = self.replay.sample(self.batch_steps, self.rng)
        with torch.no_grad():
            y = rs + self.cfg.gamma * (1 - d) * self._q(self.critic_t, o2, self._pi(self.actor_t, o2))
        lc = nn.functional.mse_loss(self._q(self.critic, o, a), y)
        self.opt_c.zero_grad(); lc.backward(); self.opt_c.step()
        la = -self._q(self.critic, o, self._pi(self.actor, o)).mean()
        self.opt_a.zero_grad(); la.backward(); self.opt_a.step()
        soft_update(self.actor_t, self.actor, self.cfg.tau)
        soft_update(self.critic_t, self.critic, self.cfg.tau)


class SharedMADDPG(_Base):
    batch_steps = 32  # x N cells per sampled step

    def __init__(self, n, f, nbr, cfg=TrainConfig(), hidden=128):
        super().__init__(n, f, nbr, cfg)
        self.actor, self.actor_t = mlp(f, A, hidden), mlp(f, A, hidden)
        cin = 2 * f + 2 * A
        self.critic, self.critic_t = mlp(cin, 1, hidden), mlp(cin, 1, hidden)
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.critic_t.load_state_dict(self.critic.state_dict())
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

    def _q(self, net, o, a):  # [B, N, F], [B, N, A] -> [B, N]
        no, na = self.M @ o, self.M @ a
        return net(torch.cat([o, a, no, na], dim=-1)).squeeze(-1)

    def act(self, obs, sigma):
        with torch.no_grad():
            logits = self.actor(torch.from_numpy(obs))
        return self._noisy(logits, sigma)

    def update(self):
        o, a, r, _, o2, d = self.replay.sample(self.batch_steps, self.rng)
        with torch.no_grad():
            a2 = torch.softmax(self.actor_t(o2), dim=-1)
            y = r + self.cfg.gamma * (1 - d[:, None]) * self._q(self.critic_t, o2, a2)
        lc = nn.functional.mse_loss(self._q(self.critic, o, a), y)
        self.opt_c.zero_grad(); lc.backward(); self.opt_c.step()
        la = -self._q(self.critic, o, torch.softmax(self.actor(o), dim=-1)).mean()
        self.opt_a.zero_grad(); la.backward(); self.opt_a.step()
        soft_update(self.actor_t, self.actor, self.cfg.tau)
        soft_update(self.critic_t, self.critic, self.cfg.tau)


def n_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
