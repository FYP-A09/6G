"""The long-trained, validation-selected MARL (marl_v2_best.py) at other network sizes.

Same protocol as marl_v2_best.py: days 1-40 for training, validation days 41-44 pick the checkpoint,
the held-out week 16-22 Dec is the test, TGNN v2 forecast as input, 4000 episodes. The reference is the
expert rule tuned on the training days (marl_v2_scaling_refs.json), scored on the same week.

Run: python src/unified/marl_v2_best_size.py train <size> <seed> [episodes]   (size 3 -> 9 cells, 10 -> 100 cells)
     python src/unified/marl_v2_best_size.py report <size>
Writes reports/unified/marl_v2_best_<cells>.json
"""
from __future__ import annotations

import copy
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
from unified.env import run_policy  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CK = ROOT / "data" / "processed" / "unified"
VAL_EVERY, VAL_DAYS = 250, (41, 45)


def week(env, pol, k=7):
    return [run_policy(env, pol, 144 * i + 1, m.DAY)["mean_reward"] for i in range(k)]


def train(size, seed, episodes):
    torch.set_num_threads(3)
    D, F = sc.build_data100()
    env = sc.make_env(D, F, size, m.TRAIN_DAYS[0] * m.DAY, m.TRAIN_DAYS[1] * m.DAY, seed)
    val = sc.make_env(D, F, size, VAL_DAYS[0] * m.DAY - 1, VAL_DAYS[1] * m.DAY)
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
            print(f"N={env.N} seed {seed} ep {ep + 1}: validation {v:.4f} (best {best:.4f}) {time.time() - t0:.0f}s", flush=True)
    torch.save({"actor": best_state, "val": best, "curve": curve, "n": env.N, "f": env.obs_dim, "episodes": episodes}, CK / f"marl_v2_best_n{env.N}_seed{seed}.pt")
    print("saved", env.N, seed)


def report(size):
    D, F = sc.build_data100()
    n = size * size
    base, end = m.TEST_DAYS[0] * m.DAY - 1, m.TEST_DAYS[1] * m.DAY
    test = sc.make_env(D, F, size, base, end)
    ref = json.loads((ROOT / "reports" / "unified" / "marl_v2_scaling_refs.json").read_text(encoding="utf-8"))[str(n)]
    tuned = np.array(ref["expert_forecast_tuned"])
    out = {"cells": n, "tuned_expert_forecast_per_day": ref["expert_forecast_tuned"], "seeds": {}}
    actors = []
    for f in sorted(CK.glob(f"marl_v2_best_n{n}_seed*.pt")):
        ck = torch.load(f, weights_only=False)
        a = SharedMADDPG(ck["n"], ck["f"], test.nbr_matrix, TrainConfig())
        a.actor.load_state_dict(ck["actor"])
        actors.append(a.actor)
        per = week(test, a.policy)
        g = np.array(per) - tuned
        out["seeds"][f.stem.split("seed")[-1]] = {"episodes": ck.get("episodes"), "validation": round(ck["val"], 4), "test_per_day": [round(x, 4) for x in per],
                                                  "test_mean": round(float(np.mean(per)), 4), "gain_over_tuned": round(float(g.mean()), 4), "days_positive": int((g > 0).sum())}
        print(n, f.stem, "validation", round(ck["val"], 4), "test", out["seeds"][f.stem.split("seed")[-1]]["test_mean"], "gain", round(float(g.mean()), 4), flush=True)

    def ens(env, obs):
        with torch.no_grad():
            lg = torch.stack([net(torch.from_numpy(obs)) for net in actors]).mean(0)
        return torch.softmax(lg, dim=-1).numpy()
    per = week(test, ens)
    g = np.array(per) - tuned
    out["ensemble"] = {"test_per_day": [round(x, 4) for x in per], "test_mean": round(float(np.mean(per)), 4), "gain_over_tuned": round(float(g.mean()), 4),
                       "ci95": round(float(2.447 * g.std(ddof=1) / 7 ** 0.5), 4), "days_positive": int((g > 0).sum()), "actors": len(actors)}
    print(n, "ensemble", out["ensemble"], flush=True)
    (ROOT / "reports" / "unified" / f"marl_v2_best_{n}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    train(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 4000) if sys.argv[1] == "train" else report(int(sys.argv[2]))
