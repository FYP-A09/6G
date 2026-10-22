"""The best shared-actor MARL we can train: longer, with checkpoint selection on validation days.

Same data and test as marl_v2.py (25 cells, TGNN v2 forecast, days 1-40 for training, held-out week
16-22 Dec for the test) with three changes: 4000 episodes instead of 2000, the checkpoint is chosen
on validation days 41-44 (never on the test week), and the final policy can be the average of
several seeds' actors (ensemble in logit space).

Run: python src/unified/marl_v2_best.py train <seed>     (saves the selected actor)
     python src/unified/marl_v2_best.py report            (tests each seed and the ensemble)
Writes reports/unified/marl_v2_best.json
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
from unified.agents import SharedMADDPG, TrainConfig  # noqa: E402
from unified.env import run_policy  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EPISODES, VAL_EVERY, VAL_DAYS = 4000, 250, (41, 45)
CK = ROOT / "data" / "processed" / "unified"


def week(env, pol, k=7):
    return [run_policy(env, pol, 144 * i + 1, m.DAY)["mean_reward"] for i in range(k)]


def train(seed):
    torch.set_num_threads(4)
    D, F = m.build_data()
    env = m.make_env(D, F, m.TRAIN_DAYS[0] * m.DAY, m.TRAIN_DAYS[1] * m.DAY, seed)
    val = m.make_env(D, F, VAL_DAYS[0] * m.DAY - 1, VAL_DAYS[1] * m.DAY)
    cfg = TrainConfig(episodes=EPISODES, seed=seed)
    agent = SharedMADDPG(env.N, env.obs_dim, env.nbr_matrix, cfg)
    best, best_state, curve, steps, t0 = -1.0, None, [], 0, time.time()
    for ep in range(EPISODES):
        sigma = cfg.noise_start + (cfg.noise_end - cfg.noise_start) * ep / (EPISODES - 1)
        obs, done = env.reset(), False
        while not done:
            act = agent.act(obs, sigma)
            obs2, r, team, done, _ = env.step(act)
            agent.replay.add(obs, act, team, r.mean(), obs2, float(done))
            obs = obs2; steps += 1
            if steps > cfg.warmup_steps:
                agent.update()
        if (ep + 1) % VAL_EVERY == 0:
            v = float(np.mean(week(val, agent.policy, 4)))
            curve.append((ep + 1, round(v, 4)))
            if v > best:
                best, best_state = v, copy.deepcopy(agent.actor.state_dict())
            print(f"seed {seed} ep {ep + 1}: validation reward {v:.4f} (best {best:.4f}) {time.time() - t0:.0f}s", flush=True)
    torch.save({"actor": best_state, "val_reward": best, "curve": curve, "n": env.N, "f": env.obs_dim}, CK / f"marl_v2_best_seed{seed}.pt")
    print("saved seed", seed)


def report():
    D, F = m.build_data()
    base, end = m.TEST_DAYS[0] * m.DAY - 1, m.TEST_DAYS[1] * m.DAY
    test = m.make_env(D, F, base, end)
    actors, out = [], {"episodes": EPISODES, "seeds": {}}
    for seed in m.SEEDS:
        ck = torch.load(CK / f"marl_v2_best_seed{seed}.pt", weights_only=False)
        a = SharedMADDPG(ck["n"], ck["f"], test.nbr_matrix, TrainConfig(seed=seed)); a.actor.load_state_dict(ck["actor"]); actors.append(a.actor)
        per = week(test, a.policy)
        out["seeds"][str(seed)] = {"validation": round(ck["val_reward"], 4), "test_per_day": [round(x, 4) for x in per], "test_mean": round(float(np.mean(per)), 4), "val_curve": ck["curve"]}
        print("seed", seed, "validation", round(ck["val_reward"], 4), "test", out["seeds"][str(seed)]["test_mean"], flush=True)

    def ens(env, obs):
        with torch.no_grad():
            logits = torch.stack([net(torch.from_numpy(obs)) for net in actors]).mean(0)
        return torch.softmax(logits, dim=-1).numpy()
    per = week(test, ens)
    out["ensemble"] = {"test_per_day": [round(x, 4) for x in per], "test_mean": round(float(np.mean(per)), 4)}
    print("ensemble test", out["ensemble"]["test_mean"])
    (ROOT / "reports" / "unified" / "marl_v2_best.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    train(int(sys.argv[2])) if sys.argv[1] == "train" else report()
