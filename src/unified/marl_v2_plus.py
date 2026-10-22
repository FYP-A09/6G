"""Pushing the shared-actor MARL past the tuned expert rule (25 cells, held-out week).

Variants, all on the marl_v2.py protocol (days 1-40 train, validation days 41-44 pick the checkpoint,
held-out week 16-22 Dec test, TGNN v2 forecast), 2000 episodes, hidden size 256, cosine learning-rate decay:

  comm      the actor also sees the mean observation of its grid neighbours (cells exchange measurements)
  res       residual policy: logits = log(tuned expert shares) + learned correction (last layer starts at
            zero, so training starts from the tuned expert and can only move away if the critic says so)
  comm_res  both

The expert's URLLC and mMTC utilisation targets (0.4, 0.4) were tuned on the training days
(marl_v2_tuned_expert.json). For the residual variants the checkpoint at episode 0 is the tuned expert itself.

Run: python src/unified/marl_v2_plus.py <variant> <seed>   -> data/processed/unified/marl_v2_plus_<variant>_<seed>.pt
     python src/unified/marl_v2_plus.py report              -> reports/unified/marl_v2_plus.json
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from unified import marl_v2 as m  # noqa: E402
from unified.agents import SharedMADDPG, TrainConfig, mlp, soft_update  # noqa: E402
from unified.env import ModelConfig, run_policy  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CK = ROOT / "data" / "processed" / "unified"
EPISODES, VAL_EVERY, VAL_DAYS, UTIL = 2000, 250, (41, 45), (0.65, 0.4, 0.4)
VARIANTS = {"comm": dict(comm=True, res=False), "res": dict(comm=False, res=True), "comm_res": dict(comm=True, res=True)}


def expert_logits(o: torch.Tensor, cfg: ModelConfig) -> torch.Tensor:
    C = cfg.capacity_mbps
    d = o[..., 14:17] * C
    cap = torch.clamp(C / (1.0 + cfg.coupling * o[..., 11]), min=1e-6)
    need = d / torch.tensor(UTIL) / cap[..., None]
    urllc = torch.clamp(need[..., 1], max=1.0)
    mmtc = torch.minimum(need[..., 2], 1.0 - urllc)
    shares = torch.stack([1.0 - urllc - mmtc, urllc, mmtc, torch.zeros_like(urllc)], dim=-1)
    return torch.log(shares.clamp(min=1e-4))


class PlusMADDPG(SharedMADDPG):
    def __init__(self, n, f, nbr, cfg, comm, res, hidden=256):
        super().__init__(n, f, nbr, cfg, hidden=hidden)
        self.comm, self.res, self.mcfg = comm, res, ModelConfig()
        din = f * (2 if comm else 1)
        self.actor, self.actor_t = mlp(din, 4, hidden), mlp(din, 4, hidden)
        if res:
            nn.init.zeros_(self.actor[-1].weight)
            nn.init.zeros_(self.actor[-1].bias)
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)

    def logits(self, net, o):
        x = torch.cat([o, self.M @ o], dim=-1) if self.comm else o
        z = net(x)
        return z + expert_logits(o, self.mcfg) if self.res else z

    def act(self, obs, sigma):
        with torch.no_grad():
            lg = self.logits(self.actor, torch.from_numpy(obs))
        return self._noisy(lg, sigma)

    def update(self):
        o, a, r, _, o2, d = self.replay.sample(self.batch_steps, self.rng)
        with torch.no_grad():
            a2 = torch.softmax(self.logits(self.actor_t, o2), dim=-1)
            y = r + self.cfg.gamma * (1 - d[:, None]) * self._q(self.critic_t, o2, a2)
        lc = nn.functional.mse_loss(self._q(self.critic, o, a), y)
        self.opt_c.zero_grad(); lc.backward(); self.opt_c.step()
        la = -self._q(self.critic, o, torch.softmax(self.logits(self.actor, o), dim=-1)).mean()
        self.opt_a.zero_grad(); la.backward(); self.opt_a.step()
        soft_update(self.actor_t, self.actor, self.cfg.tau)
        soft_update(self.critic_t, self.critic, self.cfg.tau)


def week(env, pol, k=7):
    return [run_policy(env, pol, 144 * i + 1, m.DAY)["mean_reward"] for i in range(k)]


def build(variant, env, seed):
    cfg = TrainConfig(episodes=EPISODES, seed=seed, noise_start=0.5)
    return PlusMADDPG(env.N, env.obs_dim, env.nbr_matrix, cfg, **VARIANTS[variant])


def train(variant, seed):
    torch.set_num_threads(3)
    D, F = m.build_data()
    env = m.make_env(D, F, m.TRAIN_DAYS[0] * m.DAY, m.TRAIN_DAYS[1] * m.DAY, seed)
    val = m.make_env(D, F, VAL_DAYS[0] * m.DAY - 1, VAL_DAYS[1] * m.DAY)
    agent = build(variant, env, seed)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(agent.opt_a, T_max=EPISODES * 36, eta_min=3e-5)
    cfg, steps, t0, curve = agent.cfg, 0, time.time(), []
    best = float(np.mean(week(val, agent.policy, 4))) if VARIANTS[variant]["res"] else -1.0
    best_state = copy.deepcopy(agent.actor.state_dict())
    curve.append((0, round(best, 4)))
    for ep in range(EPISODES):
        sigma = cfg.noise_start + (cfg.noise_end - cfg.noise_start) * ep / (EPISODES - 1)
        obs, done = env.reset(), False
        while not done:
            act = agent.act(obs, sigma)
            obs2, r, team, done, _ = env.step(act)
            agent.replay.add(obs, act, team, r.mean(), obs2, float(done))
            obs = obs2
            steps += 1
            if steps > cfg.warmup_steps:
                agent.update()
                sched.step()
        if (ep + 1) % VAL_EVERY == 0:
            v = float(np.mean(week(val, agent.policy, 4)))
            curve.append((ep + 1, round(v, 4)))
            if v > best:
                best, best_state = v, copy.deepcopy(agent.actor.state_dict())
            print(f"{variant} seed {seed} ep {ep + 1}: validation {v:.4f} (best {best:.4f}) {time.time() - t0:.0f}s", flush=True)
    torch.save({"actor": best_state, "val": best, "curve": curve, "n": env.N, "f": env.obs_dim}, CK / f"marl_v2_plus_{variant}_{seed}.pt")
    print("saved", variant, seed)


def report():
    D, F = m.build_data()
    base, end = m.TEST_DAYS[0] * m.DAY - 1, m.TEST_DAYS[1] * m.DAY
    test = m.make_env(D, F, base, end)
    ref = json.loads((ROOT / "reports" / "unified" / "marl_v2_tuned_expert.json").read_text(encoding="utf-8"))["forecast"]["test_per_day"]
    out = {"tuned_expert_forecast_per_day": ref, "variants": {}}
    for f in sorted(CK.glob("marl_v2_plus_*.pt")):
        variant, seed = f.stem.replace("marl_v2_plus_", "").rsplit("_", 1)
        ck = torch.load(f, weights_only=False)
        a = build(variant, test, int(seed))
        a.actor.load_state_dict(ck["actor"])
        per = week(test, a.policy)
        gain = float(np.mean(np.array(per) - np.array(ref)))
        out["variants"].setdefault(variant, {})[seed] = {"validation": round(ck["val"], 4), "test_per_day": [round(x, 4) for x in per],
                                                         "test_mean": round(float(np.mean(per)), 4), "paired_gain_over_tuned_expert": round(gain, 4), "val_curve": ck["curve"]}
        print(variant, seed, "val", round(ck["val"], 4), "test", out["variants"][variant][seed]["test_mean"], "vs tuned expert", round(gain, 4), flush=True)
    (ROOT / "reports" / "unified" / "marl_v2_plus.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    report() if sys.argv[1] == "report" else train(sys.argv[1], int(sys.argv[2]))
