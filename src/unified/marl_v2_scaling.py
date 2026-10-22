"""Shared actor vs a single centralised agent, trained properly, at 9 / 25 / 100 cells.

Same protocol as marl_v2.py: days 1-40 for training, the held-out week (days 45-51) for the test,
the v2 forecast as input, 1000 episodes for every learner. The single agent is run with two
actor learning rates (1e-4, the earlier sweep's best, and the MADDPG default 3e-4) and the better
of the two is reported, so it is not handicapped by the learning rate.

Run: python src/unified/marl_v2_scaling.py <size>   (3, 5 or 10 -> 9, 25 or 100 cells)
Writes reports/unified/marl_v2_scaling_<cells>.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from unified import forecast_city_v2 as v2  # noqa: E402
from unified import marl_v2 as m  # noqa: E402
from unified.agents import SingleAgentDDPG, SharedMADDPG, TrainConfig, n_params  # noqa: E402
from unified.data import GRID, block_cells, grid_neighbours, load_block_activity  # noqa: E402
from unified.env import ModelConfig, UnifiedSlicingEnv, demand_scale, run_policy  # noqa: E402
from unified.showcase import MILAN  # noqa: E402
from unified.twin_dataset import BASE, full_grid  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CACHE100 = ROOT / "data" / "processed" / "unified" / "marl_v2_data100.npz"
EPISODES, SEEDS = 1000, (0, 1)


def build_data100():
    if CACHE100.exists():
        z = np.load(CACHE100)
        return z["demand"], z["forecast"]
    cfg = ModelConfig()
    block = load_block_activity(MILAN)
    scale = demand_scale(block["activity"][:720], cfg).astype(np.float32)
    ck = torch.load(v2.MODEL, weights_only=False)
    model = v2.CityTGNN(); model.load_state_dict(ck["state"]); model.eval()
    cs = ck["cell_scale"]
    grid = full_grid(BASE / "milan_full_grid_62d.npy")
    H, lo = m.HALO, 44
    cells = np.array([r * GRID + c for r in range(lo, lo + H) for c in range(lo, lo + H)])
    cs_h, T = cs[cells], m.TEST_DAYS[1] * m.DAY
    D = np.zeros((T, H * H, 3), np.float32)
    for a in range(0, T, m.DAY):
        D[a:a + m.DAY] = np.asarray(grid[a:a + m.DAY])[:, cells] * scale
    ei = v2.edge_index_for(range(H), range(H))
    inner = np.array([r * H + c for r in range(1, H - 1) for c in range(1, H - 1)])
    F = D.copy()
    with torch.no_grad():
        for t in range(m.DAY + m.WINDOW, T - 1):
            F[t] = model(v2.make_window(D, t, np.arange(H * H), cs_h, 0), ei).numpy() * cs_h
    np.savez_compressed(CACHE100, demand=D[:, inner], forecast=F[:, inner])
    return D[:, inner], F[:, inner]


def make_env(D, F, size, a, b, seed=0):
    full = block_cells(10)
    idx = [full.index(c) for c in block_cells(size)]
    return UnifiedSlicingEnv(D[a:b][:, idx], grid_neighbours(block_cells(size)), ModelConfig(), episode_len=36,
                             start_range=(0, b - a - 41), forecast=F[a:b][:, idx], seed=seed)


def main(size):
    torch.set_num_threads(4)
    D, F = build_data100()
    base, end = m.TEST_DAYS[0] * m.DAY - 1, m.TEST_DAYS[1] * m.DAY
    n = size * size
    out = {"cells": n, "episodes": EPISODES, "runs": []}
    configs = [("marl", None), ("single_lr1e-4", 1e-4), ("single_lr3e-4", 3e-4)]
    for seed in SEEDS:
        for name, lr in configs:
            t0 = time.time()
            env = make_env(D, F, size, m.TRAIN_DAYS[0] * m.DAY, m.TRAIN_DAYS[1] * m.DAY, seed)
            test = make_env(D, F, size, base, end)
            cfg = TrainConfig(episodes=EPISODES, seed=seed)
            if lr is not None:
                cfg.actor_lr = lr
            agent = (SharedMADDPG if lr is None else SingleAgentDDPG)(env.N, env.obs_dim, env.nbr_matrix, cfg)
            hist = agent.train(env, log_every=500)
            per_day = [run_policy(test, agent.policy, 144 * k + 1, m.DAY)["mean_reward"] for k in range(7)]
            out["runs"].append({"config": name, "seed": seed, "mean": round(float(np.mean(per_day)), 4), "per_day": [round(x, 4) for x in per_day],
                                "actor_params": n_params(agent.actor), "train_seconds": round(time.time() - t0),
                                "last_100_train_reward": round(float(np.mean(hist["episode_reward"][-100:])), 4)})
            print(f"N={n} {name} seed {seed}: held-out week {out['runs'][-1]['mean']} ({out['runs'][-1]['train_seconds']}s)", flush=True)
            (ROOT / "reports" / "unified" / f"marl_v2_scaling_{n}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", n)


if __name__ == "__main__":
    main(int(sys.argv[1]))
