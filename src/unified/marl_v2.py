"""Does MARL benefit once it is trained properly? (25 cells, held-out week)

The first MARL study trained on 5 days for 300 episodes, and its learning curves were
still rising. Here every learned controller gets:
  data      days 1-40 of real Milan traffic (not 5), forecast = TGNN v2
  budget    2000 episodes of 6 hours (about 72,000 environment steps)
  test      the held-out week 16-22 Dec (days 45-51); one reward per day

Variants (3 seeds each, same data, budget and test):
  ddpg_fc       the original shared-actor MADDPG with the forecast as input
  bc_fc         shared actor + neighbourhood critic, actor loss in the TD3+BC form
                (Fujimoto and Gu, NeurIPS 2021): maximise Q while staying close to the
                expert "headroom" rule computed from the cell's own observation (forecast)
  bc_nofc       the same, with no forecast anywhere (expert uses current demand)
References on the same week: equal split, expert on current demand, expert on the forecast,
expert on the true next demand.

Run: python src/unified/marl_v2.py <variant>      (variant in ddpg_fc, bc_fc, bc_nofc, refs)
Writes reports/unified/marl_v2_<variant>.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from unified import forecast_city_v2 as v2  # noqa: E402
from unified.agents import SharedMADDPG, TrainConfig, n_params, soft_update  # noqa: E402
from unified.data import GRID, block_cells, grid_neighbours, load_block_activity  # noqa: E402
from unified.env import HEADROOM_UTILISATION, ModelConfig, UnifiedSlicingEnv, demand_scale, equal_split, headroom, run_policy  # noqa: E402
from unified.showcase import MILAN  # noqa: E402
from unified.twin_dataset import BASE, full_grid  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DAY, WINDOW = v2.DAY, v2.WINDOW
SIZE, EPISODES, SEEDS = 5, 2000, (0, 1, 2)
TRAIN_DAYS, TEST_DAYS = (1, 41), (45, 52)
CACHE = ROOT / "data" / "processed" / "unified" / "marl_v2_data.npz"
HALO = 12  # 10x10 block plus a one-cell ring


def build_data():
    """Demand and v2 forecast for the central 25 cells over days 0-51 (cached)."""
    if CACHE.exists():
        z = np.load(CACHE)
        return z["demand"], z["forecast"]
    cfg = ModelConfig()
    block = load_block_activity(MILAN)
    scale = demand_scale(block["activity"][:720], cfg).astype(np.float32)
    ck = torch.load(v2.MODEL, weights_only=False)
    model = v2.CityTGNN(); model.load_state_dict(ck["state"]); model.eval()
    cs = ck["cell_scale"]
    grid = full_grid(BASE / "milan_full_grid_62d.npy")
    lo = 44
    cells = np.array([r * GRID + c for r in range(lo, lo + HALO) for c in range(lo, lo + HALO)])
    cs_h = cs[cells]
    T = TEST_DAYS[1] * DAY
    D = np.zeros((T, HALO * HALO, 3), np.float32)
    for a in range(0, T, DAY):
        D[a:a + DAY] = np.asarray(grid[a:a + DAY])[:, cells] * scale
    ei = v2.edge_index_for(range(HALO), range(HALO))
    inner = np.array([r * HALO + c for r in range(1, HALO - 1) for c in range(1, HALO - 1)])
    # the 25 central cells of the 10x10 block, in block_cells(5) order
    full = block_cells(10)
    idx25 = np.array([inner[full.index(c)] for c in block_cells(SIZE)])
    compact = np.arange(HALO * HALO)
    F = D.copy()
    with torch.no_grad():
        for t in range(DAY + WINDOW, T - 1):
            F[t] = (model(v2.make_window(D, t, compact, cs_h, 0), ei).numpy() * cs_h)
    np.savez_compressed(CACHE, demand=D[:, idx25], forecast=F[:, idx25])
    return D[:, idx25], F[:, idx25]


def nbrs():
    return grid_neighbours(block_cells(SIZE))


def make_env(demand, forecast, a, b, seed=0):
    return UnifiedSlicingEnv(demand[a:b], nbrs(), ModelConfig(), episode_len=36, start_range=(0, b - a - 41),
                             forecast=None if forecast is None else forecast[a:b], seed=seed)


def expert_from_obs(o: torch.Tensor, use_forecast: bool, cfg: ModelConfig) -> torch.Tensor:
    """The headroom rule rebuilt from a cell's own observation (demand, forecast, neighbour load)."""
    C = cfg.capacity_mbps
    d = (o[..., 14:17] if use_forecast else o[..., 0:3]) * C
    cap = torch.clamp(C / (1.0 + cfg.coupling * o[..., 11]), min=1e-6)
    need = d / torch.tensor(HEADROOM_UTILISATION) / cap[..., None]
    urllc = torch.clamp(need[..., 1], max=1.0)
    mmtc = torch.minimum(need[..., 2], 1.0 - urllc)
    embb = 1.0 - urllc - mmtc
    return torch.stack([embb, urllc, mmtc, torch.zeros_like(embb)], dim=-1)


class BCMADDPG(SharedMADDPG):
    """Shared actor and neighbourhood critic; actor loss = -lambda * Q + MSE to the expert (TD3+BC form)."""

    def __init__(self, n, f, nbr, cfg, use_forecast, alpha=2.5):
        super().__init__(n, f, nbr, cfg)
        self.use_forecast, self.alpha, self.model_cfg = use_forecast, alpha, ModelConfig()

    def update(self):
        o, a, r, _, o2, d = self.replay.sample(self.batch_steps, self.rng)
        with torch.no_grad():
            a2 = torch.softmax(self.actor_t(o2), dim=-1)
            y = r + self.cfg.gamma * (1 - d[:, None]) * self._q(self.critic_t, o2, a2)
        lc = nn.functional.mse_loss(self._q(self.critic, o, a), y)
        self.opt_c.zero_grad(); lc.backward(); self.opt_c.step()
        pi = torch.softmax(self.actor(o), dim=-1)
        q = self._q(self.critic, o, pi)
        lam = self.alpha / q.abs().mean().detach().clamp(min=1e-6)
        la = -lam * q.mean() + nn.functional.mse_loss(pi, expert_from_obs(o, self.use_forecast, self.model_cfg))
        self.opt_a.zero_grad(); la.backward(); self.opt_a.step()
        soft_update(self.actor_t, self.actor, self.cfg.tau)
        soft_update(self.critic_t, self.critic, self.cfg.tau)


def week_rewards(env_test, policy):
    return [run_policy(env_test, policy, 144 * k + 1, DAY)["mean_reward"] for k in range(7)]


def ci(x):
    x = np.asarray(x, float)
    return round(float(x.mean()), 4), round(float({2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447}.get(len(x), 1.96) * x.std(ddof=1) / np.sqrt(len(x))), 4)


def main(variant: str):
    torch.set_num_threads(4)
    D, F = build_data()
    base = TEST_DAYS[0] * DAY - 1
    end = TEST_DAYS[1] * DAY
    out = {"variant": variant, "cells": SIZE * SIZE, "train_days": list(TRAIN_DAYS), "test_days": list(TEST_DAYS), "episodes": EPISODES}
    if variant == "refs":
        env = make_env(D, F, base, end)
        refs = {"equal_split": lambda e, o: equal_split(e, o), "expert_current": lambda e, o: headroom(e, o, "current"),
                "expert_forecast": lambda e, o: headroom(e, o, "forecast"), "expert_oracle": lambda e, o: headroom(e, o, "oracle")}
        out["policies"] = {k: {"per_day": [round(x, 4) for x in week_rewards(env, p)]} for k, p in refs.items()}
    else:
        use_fc = variant != "bc_nofc"
        runs = []
        for seed in SEEDS:
            t0 = time.time()
            env = make_env(D, F if use_fc else None, TRAIN_DAYS[0] * DAY, TRAIN_DAYS[1] * DAY, seed)
            test_env = make_env(D, F if use_fc else None, base, end)
            cfg = TrainConfig(episodes=EPISODES, seed=seed, noise_start=1.0 if variant == "ddpg_fc" else 0.5)
            agent = SharedMADDPG(env.N, env.obs_dim, env.nbr_matrix, cfg) if variant == "ddpg_fc" else BCMADDPG(env.N, env.obs_dim, env.nbr_matrix, cfg, use_fc)
            hist = agent.train(env, log_every=250)
            per_day = week_rewards(test_env, agent.policy)
            runs.append({"seed": seed, "per_day": [round(x, 4) for x in per_day], "mean": round(float(np.mean(per_day)), 4),
                         "train_seconds": round(time.time() - t0), "actor_params": n_params(agent.actor),
                         "learning_curve": [round(float(np.mean(hist["episode_reward"][i:i + 50])), 4) for i in range(0, EPISODES, 50)]})
            print(f"{variant} seed {seed}: held-out week reward {runs[-1]['mean']}", flush=True)
        out["runs"] = runs
    (ROOT / "reports" / "unified" / f"marl_v2_{variant}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", variant)


if __name__ == "__main__":
    main(sys.argv[1])
