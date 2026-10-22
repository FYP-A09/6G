"""MARL that sees more than the expert rule can use: recent history and yesterday's traffic.

The expert headroom rule uses one number per slice (current demand, or the one-step forecast). The learned actor
can use more, so here each cell also observes, as extra inputs: its own demand 10 and 30 minutes ago, its demand at
the same time yesterday for the coming interval (all known to a real controller), the forecast error of the previous
interval, and a weekend flag. Everything else is as in marl_v2_best.py (days 1-40 train, validation days 41-44 pick
the checkpoint, held-out week 16-22 Dec test, 4000 episodes, 25 cells).

Run: python src/unified/marl_v2_richobs.py train <seed> [episodes] ; python src/unified/marl_v2_richobs.py report
Writes reports/unified/marl_v2_richobs.json
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from unified import marl_v2 as m  # noqa: E402
from unified import marl_v2_scaling as sc  # noqa: E402
from unified.agents import SharedMADDPG, TrainConfig  # noqa: E402
from unified.data import block_cells, grid_neighbours  # noqa: E402
from unified.env import ModelConfig, UnifiedSlicingEnv, run_policy  # noqa: E402
from unified.twin_dataset import START  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CK = ROOT / "data" / "processed" / "unified"
SIZE, VAL_EVERY, VAL_DAYS = 5, 250, (41, 45)
K_EXTRA = 3 * 3 + 3 + 1


class ExtraObsEnv(UnifiedSlicingEnv):
    def __init__(self, *a, extras, **kw):
        super().__init__(*a, **kw)
        self.extras = extras
        self.obs_dim += extras.shape[-1]

    def _obs(self):
        base = super()._obs()
        return np.concatenate([base, self.extras[self.t]], axis=1).astype(np.float32)


def build(D, F, a, b, seed=0):
    """Environment over absolute intervals [a, b) with the extra observations computed from the full history."""
    full = block_cells(10)
    idx = [full.index(c) for c in block_cells(SIZE)]
    d, f = D[:, idx], F[:, idx]
    C = ModelConfig().capacity_mbps
    t = np.arange(a, b)
    lag = lambda k: d[np.clip(t - k, 0, None)] / C
    yday = d[np.clip(t + 1 - 144, 0, None)] / C
    err = (f[np.clip(t - 1, 0, None)] - d[t]) / C                       # forecast made one interval ago for now, minus truth
    wk = np.array([(START + dt.timedelta(int(x + 1) // 144)).weekday() >= 5 for x in t], np.float32)[:, None, None].repeat(len(idx), 1)
    extras = np.concatenate([lag(1), lag(3), yday, err, wk], axis=2).astype(np.float32)
    return ExtraObsEnv(d[a:b], grid_neighbours(block_cells(SIZE)), ModelConfig(), episode_len=36, start_range=(0, b - a - 41),
                       forecast=f[a:b], seed=seed, extras=extras)


def week(env, pol, k=7):
    return [run_policy(env, pol, 144 * i + 1, m.DAY)["mean_reward"] for i in range(k)]


def train(seed, episodes):
    torch.set_num_threads(3)
    D, F = sc.build_data100()
    env = build(D, F, m.TRAIN_DAYS[0] * m.DAY, m.TRAIN_DAYS[1] * m.DAY, seed)
    val = build(D, F, VAL_DAYS[0] * m.DAY - 1, VAL_DAYS[1] * m.DAY)
    cfg = TrainConfig(episodes=episodes, seed=seed)
    agent = SharedMADDPG(env.N, env.obs_dim, env.nbr_matrix, cfg)
    best, best_state, curve, steps, t0 = -1.0, None, [], 0, time.time()
    for ep in range(episodes):
        sigma = cfg.noise_start + (cfg.noise_end - cfg.noise_start) * ep / (episodes - 1)
        obs, done = env.reset(), False
        while not done:
            act = agent.act(obs, sigma)
            obs2, r, team, done, _ = env.step(act)
            agent.replay.add(obs, act, team, r.mean(), obs2, float(done))
            obs = obs2
            steps += 1
            if steps > cfg.warmup_steps:
                agent.update()
        if (ep + 1) % VAL_EVERY == 0:
            v = float(np.mean(week(val, agent.policy, 4)))
            curve.append((ep + 1, round(v, 4)))
            if v > best:
                best, best_state = v, copy.deepcopy(agent.actor.state_dict())
            print(f"richobs seed {seed} ep {ep + 1}: validation {v:.4f} (best {best:.4f}) {time.time() - t0:.0f}s", flush=True)
    torch.save({"actor": best_state, "val": best, "curve": curve, "n": env.N, "f": env.obs_dim}, CK / f"marl_v2_richobs_seed{seed}.pt")
    print("saved", seed)


def report():
    D, F = sc.build_data100()
    base, end = m.TEST_DAYS[0] * m.DAY - 1, m.TEST_DAYS[1] * m.DAY
    test = build(D, F, base, end)
    rich = json.loads((ROOT / "reports" / "unified" / "marl_v2_tuned_expert_rich.json").read_text(encoding="utf-8"))["25"]["forecast"]["test_per_day"]
    simple = json.loads((ROOT / "reports" / "unified" / "marl_v2_tuned_expert.json").read_text(encoding="utf-8"))["forecast"]["test_per_day"]
    out, actors = {"rich_expert_per_day": rich, "seeds": {}}, []
    for f in sorted(CK.glob("marl_v2_richobs_seed*.pt")):
        ck = torch.load(f, weights_only=False)
        a = SharedMADDPG(ck["n"], ck["f"], test.nbr_matrix, TrainConfig())
        a.actor.load_state_dict(ck["actor"])
        actors.append(a.actor)
        per = week(test, a.policy)
        g = np.array(per) - np.array(rich)
        out["seeds"][f.stem.split("seed")[-1]] = {"validation": round(ck["val"], 4), "test_per_day": [round(x, 4) for x in per], "test_mean": round(float(np.mean(per)), 4),
                                                  "gain_over_rich_expert": round(float(g.mean()), 4), "days_positive": int((g > 0).sum()), "val_curve": ck["curve"]}
        print(f.stem, "val", round(ck["val"], 4), "test", out["seeds"][f.stem.split("seed")[-1]]["test_mean"], "vs rich expert", round(float(g.mean()), 4), int((g > 0).sum()), "of 7", flush=True)

    def ens(env, obs):
        with torch.no_grad():
            lg = torch.stack([net(torch.from_numpy(obs)) for net in actors]).mean(0)
        return torch.softmax(lg, dim=-1).numpy()
    per = week(test, ens)
    g = np.array(per) - np.array(rich)
    out["ensemble"] = {"test_per_day": [round(x, 4) for x in per], "test_mean": round(float(np.mean(per)), 4), "gain_over_rich_expert": round(float(g.mean()), 4),
                       "ci95": round(float(2.447 * g.std(ddof=1) / 7 ** 0.5), 4), "days_positive": int((g > 0).sum()),
                       "gain_over_simple_expert": round(float((np.array(per) - np.array(simple)).mean()), 4)}
    print("ensemble", out["ensemble"], flush=True)
    (ROOT / "reports" / "unified" / "marl_v2_richobs.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    train(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 4000) if sys.argv[1] == "train" else report()
