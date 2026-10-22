"""PPO-family MARL baselines on the unified environment: IPPO and MAPPO-style (shared parameters).

  ippo    each cell's actor and value function see only that cell's own observation
  mappo   the value function also sees the mean observation of the cell's grid neighbours (the neighbourhood
          form of a centralised critic, as in marl_v2's DDPG critic); the actor stays local

Same protocol as marl_v2_best.py (25 cells, days 1-40, TGNN v2 forecast, validation days 41-44 choose the checkpoint,
held-out week test) with 8000 episodes because on-policy learners are less sample efficient. The policy is a Gaussian
over the four slice logits (softmax gives the shares); advantages use GAE(lambda 0.95, gamma 0.9) on the team reward;
clip 0.2, 4 epochs, 10 episodes per update, entropy bonus 0.003.

Run: python src/unified/marl_v2_ppo.py train <ippo|mappo> <seed> ; python src/unified/marl_v2_ppo.py report
Writes reports/unified/marl_v2_ppo.json
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
from unified.agents import mlp  # noqa: E402
from unified.env import run_policy  # noqa: E402
from unified import qos_alt  # noqa: E402

SUFFIX = "_simu5g" if qos_alt.maybe_apply() else ""   # QOS=simu5g trains and tests on the Simu5G-shaped latency curve

ROOT = Path(__file__).resolve().parents[2]
CK = ROOT / "data" / "processed" / "unified"
EPISODES, VAL_EVERY, VAL_DAYS, EP_PER_UPDATE = 8000, 500, (41, 45), 10
GAMMA, LAM, CLIP, EPOCHS, ENT = 0.9, 0.95, 0.2, 4, 0.003


class Actor(nn.Module):
    def __init__(self, f, hidden=128):
        super().__init__()
        self.mean = mlp(f, 4, hidden)
        self.log_std = nn.Parameter(torch.full((4,), -0.5))

    def dist(self, o):
        return torch.distributions.Normal(self.mean(o), self.log_std.exp())


class PPO:
    def __init__(self, n, f, nbr, mode, seed):
        torch.manual_seed(seed)
        self.mode, self.M = mode, torch.from_numpy(nbr.astype(np.float32))
        self.actor = Actor(f)
        self.value = mlp(f * (2 if mode == "mappo" else 1), 1, 128)
        self.opt = torch.optim.Adam(list(self.actor.parameters()) + list(self.value.parameters()), lr=3e-4)

    def v(self, o):
        x = torch.cat([o, self.M @ o], dim=-1) if self.mode == "mappo" else o
        return self.value(x).squeeze(-1)

    def policy(self, env, obs):
        with torch.no_grad():
            return torch.softmax(self.actor.mean(torch.from_numpy(obs)), dim=-1).numpy()

    def rollout(self, env, episodes):
        eps = []
        for _ in range(episodes):
            obs, done = env.reset(), False
            O, A, LP, R, V = [], [], [], [], []
            while not done:
                o = torch.from_numpy(obs)
                with torch.no_grad():
                    dist = self.actor.dist(o)
                    z = dist.sample()
                    lp, val = dist.log_prob(z).sum(-1), self.v(o)
                obs2, r, team, done, _ = env.step(torch.softmax(z, dim=-1).numpy())
                O.append(o); A.append(z); LP.append(lp); R.append(torch.from_numpy(team.astype(np.float32))); V.append(val)
                obs = obs2
            with torch.no_grad():
                last = self.v(torch.from_numpy(obs))          # episodes are time-limited, so bootstrap from the last state
            eps.append((torch.stack(O), torch.stack(A), torch.stack(LP), torch.stack(R), torch.stack(V), last))
        return eps

    def update(self, eps):
        Os, As, LPs, ADs, RTs = [], [], [], [], []
        for O, A, LP, R, V, last in eps:
            T = O.shape[0]
            nxt = torch.cat([V[1:], last[None]], dim=0)
            delta = R + GAMMA * nxt - V
            gae, adv = torch.zeros_like(last), torch.zeros_like(V)
            for t in reversed(range(T)):
                gae = delta[t] + GAMMA * LAM * gae
                adv[t] = gae
            Os.append(O); As.append(A); LPs.append(LP); ADs.append(adv); RTs.append(adv + V)
        O, A, LP, adv, ret = map(torch.cat, (Os, As, LPs, ADs, RTs))
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        for _ in range(EPOCHS):
            perm = torch.randperm(O.shape[0])
            for i in range(0, len(perm), 64):
                idx = perm[i:i + 64]
                o, a, lp0, ad, rt = O[idx], A[idx], LP[idx], adv[idx], ret[idx]
                dist = self.actor.dist(o)
                lp = dist.log_prob(a).sum(-1)
                ratio = (lp - lp0).exp()
                pl = -torch.min(ratio * ad, ratio.clamp(1 - CLIP, 1 + CLIP) * ad).mean()
                vl = nn.functional.mse_loss(self.v(o), rt)
                loss = pl + 0.5 * vl - ENT * dist.entropy().sum(-1).mean()
                self.opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.parameters(), 0.5); self.opt.step()

    def parameters(self):
        return list(self.actor.parameters()) + list(self.value.parameters())


def week(env, pol, k=7):
    return [run_policy(env, pol, 144 * i + 1, m.DAY)["mean_reward"] for i in range(k)]


def train(mode, seed):
    torch.set_num_threads(3)
    D, F = m.build_data()
    env = m.make_env(D, F, m.TRAIN_DAYS[0] * m.DAY, m.TRAIN_DAYS[1] * m.DAY, seed)
    val = m.make_env(D, F, VAL_DAYS[0] * m.DAY - 1, VAL_DAYS[1] * m.DAY)
    agent = PPO(env.N, env.obs_dim, env.nbr_matrix, mode, seed)
    best, best_state, curve, t0 = -1.0, None, [], time.time()
    for ep in range(0, EPISODES, EP_PER_UPDATE):
        agent.update(agent.rollout(env, EP_PER_UPDATE))
        if (ep + EP_PER_UPDATE) % VAL_EVERY == 0:
            v = float(np.mean(week(val, agent.policy, 4)))
            curve.append((ep + EP_PER_UPDATE, round(v, 4)))
            if v > best:
                best, best_state = v, copy.deepcopy(agent.actor.state_dict())
            print(f"{mode} seed {seed} ep {ep + EP_PER_UPDATE}: validation {v:.4f} (best {best:.4f}) {time.time() - t0:.0f}s", flush=True)
    torch.save({"actor": best_state, "val": best, "curve": curve, "n": env.N, "f": env.obs_dim}, CK / f"marl_v2_ppo_{mode}_{seed}{SUFFIX}.pt")
    print("saved", mode, seed)


def report():
    D, F = m.build_data()
    base, end = m.TEST_DAYS[0] * m.DAY - 1, m.TEST_DAYS[1] * m.DAY
    test = m.make_env(D, F, base, end)
    ref = {}
    if not SUFFIX:
        ref["rich"] = json.loads((ROOT / "reports" / "unified" / "marl_v2_tuned_expert_rich.json").read_text(encoding="utf-8"))["25"]["forecast"]["test_per_day"]
        ref["simple"] = json.loads((ROOT / "reports" / "unified" / "marl_v2_tuned_expert.json").read_text(encoding="utf-8"))["forecast"]["test_per_day"]
    out = {}
    for f in sorted(CK.glob("marl_v2_ppo_*.pt")):
        if (SUFFIX == "") == ("simu5g" in f.stem):
            continue
        mode, seed = f.stem.replace("marl_v2_ppo_", "").replace(SUFFIX, "").split("_")
        ck = torch.load(f, weights_only=False)
        a = PPO(ck["n"], ck["f"], test.nbr_matrix, mode, int(seed))
        a.actor.load_state_dict(ck["actor"])
        per = week(test, a.policy)
        row = {"validation": round(ck["val"], 4), "test_per_day": [round(x, 4) for x in per], "test_mean": round(float(np.mean(per)), 4), "val_curve": ck["curve"]}
        if ref:
            row["gain_over_simple_tuned_expert"] = round(float((np.array(per) - np.array(ref["simple"])).mean()), 4)
            row["gain_over_rich_expert"] = round(float((np.array(per) - np.array(ref["rich"])).mean()), 4)
        out.setdefault(mode, {})[seed] = row
        print(mode, seed, "val", row["validation"], "test", row["test_mean"], flush=True)
    (ROOT / "reports" / "unified" / f"marl_v2_ppo{SUFFIX}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    train(sys.argv[2], int(sys.argv[3])) if sys.argv[1] == "train" else report()
