"""What is the city forecaster (v2) worth to a slicing controller? Re-run of the forecast-value study.

v2 never trained on days 5 and 6 (the experiments' test days), so its forecasts there are
out of sample. Three parts:

  A  experiment days (days 5-6, the same test split as the MARL study): the expert
     "headroom" policy fed current demand / v1 forecast / v2 forecast / true demand,
     and equal split, at 9-100 cells
  B  held-out week (days 45-51, 16-22 Dec): the same policies plus "same time
     yesterday" as the forecast, one reward per day, mean and 95% CI over the 7 days,
     and the paired gain of v2 over current demand
  C  MARL (shared actor, 25 cells, 3 seeds) fed the v2 forecast, next to the existing
     no-forecast / v1 / perfect-forecast arms

Run: python src/unified/forecast_value_v2.py   ->  reports/unified/forecast_value_v2.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from unified import forecast_city_v2 as v2  # noqa: E402
from unified import experiments as ex  # noqa: E402
from unified.agents import SharedMADDPG  # noqa: E402
from unified.data import GRID, block_cells, grid_neighbours, load_block_activity  # noqa: E402
from unified.env import ModelConfig, UnifiedSlicingEnv, demand_scale, equal_split, headroom, run_policy  # noqa: E402
from unified.showcase import MILAN  # noqa: E402
from unified.twin_dataset import BASE, full_grid  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CACHE_V2 = ROOT / "data" / "processed" / "unified" / "tgnn_forecast_v2.npz"
DAY, WINDOW = v2.DAY, v2.WINDOW
LO, HI = 44, 56                       # block rows/cols 45-54 plus a one-cell halo
SIZES = (3, 5, 7, 10)


def block_forecast(model, D, t0, ts, cs):
    """Forecast of t+1 for the 10 x 10 block at each t in ts (run with a one-cell halo)."""
    cells = np.array([r * GRID + c for r in range(LO, HI) for c in range(LO, HI)])
    ei = v2.edge_index_for(range(LO, HI), range(LO, HI))
    inner = np.array([(r - LO) * (HI - LO) + (c - LO) for r in range(LO + 1, HI - 1) for c in range(LO + 1, HI - 1)])
    out = np.zeros((len(ts), 100, 3), np.float32)
    model.eval()
    with torch.no_grad():
        for k, t in enumerate(ts):
            out[k] = (model(v2.make_window(D, t, cells, cs, t0), ei).numpy() * cs[cells])[inner]
    return out


def sub_index(size):
    full = block_cells(10)
    return [full.index(c) for c in block_cells(size)]


def ci95(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def policies():
    return {"equal_split": lambda e, o: equal_split(e, o),
            "expert_current": lambda e, o: headroom(e, o, "current"),
            "expert_forecast": lambda e, o: headroom(e, o, "forecast"),
            "expert_oracle": lambda e, o: headroom(e, o, "oracle")}


def main():
    t_start = time.time()
    torch.set_num_threads(8)
    cfg = ModelConfig()
    block = load_block_activity(MILAN)
    scale = demand_scale(block["activity"][:ex.TRAIN_END], cfg)
    demand = block["activity"] * scale
    cells_full = [int(c) for c in block["cells"]]
    ck = torch.load(v2.MODEL, weights_only=False)
    model = v2.CityTGNN(); model.load_state_dict(ck["state"]); cs = ck["cell_scale"]
    grid = full_grid(BASE / "milan_full_grid_62d.npy")
    sc32 = scale.astype(np.float32)
    report = {"model": str(v2.MODEL.name), "excluded_training_days": list(v2.EXCLUDE_DAYS)}

    # ---- A. experiment days -------------------------------------------------------------
    D = v2.load_range(grid, sc32, 0, 7 * DAY)
    ts = np.arange(DAY, 7 * DAY - 1)                                   # first day has no "yesterday": persistence there
    f = np.array(demand, np.float32); f[ts] = block_forecast(model, D, 0, ts, cs)
    CACHE_V2.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE_V2, forecast=f)
    f1 = np.load(ex.CACHE)["forecast"]
    A = {}
    for size in SIZES:
        n = size * size
        row = {}
        for label, fc in (("v1_5days", f1), ("v2", f)):
            env = ex.make_env(demand, cells_full, size, fc)
            for name, pol in policies().items():
                if name in ("equal_split", "expert_current", "expert_oracle") and label == "v1_5days" and name != "expert_current":
                    continue
                r = ex.evaluate(env, pol)["mean_reward"]
                row[f"{name}" if name != "expert_forecast" else f"expert_forecast_{label}"] = round(float(r), 4)
        A[str(n)] = row
        print("A", n, row, flush=True)
    report["experiment_days"] = A

    # ---- B. held-out week ---------------------------------------------------------------
    base = 45 * DAY - 1
    t0 = 45 * DAY - DAY - WINDOW
    Dw = v2.load_range(grid, sc32, t0, 52 * DAY)
    tsw = np.arange(base, 52 * DAY - 1)
    Fw = np.zeros((52 * DAY - base, 100, 3), np.float32)
    Fw[:-1] = block_forecast(model, Dw, t0, tsw, cs); Fw[-1] = Dw[52 * DAY - 1 - t0][np.array([r * GRID + c for r in range(45, 55) for c in range(45, 55)])]
    inner_cells = np.array([r * GRID + c for r in range(45, 55) for c in range(45, 55)])
    Wd = Dw[base - t0:52 * DAY - t0][:, inner_cells]                    # true demand, [T, 100, 3]
    Yd = np.concatenate([Dw[base - DAY - t0 + 1:52 * DAY - DAY - t0 + 1][:, inner_cells]], axis=0)   # same time yesterday for t+1
    B = {}
    for size in SIZES:
        n = size * size; idx = sub_index(size); nbrs = grid_neighbours(block_cells(size))
        per = {}
        for fname, fc in (("v2", Fw), ("yesterday", Yd)):
            env = UnifiedSlicingEnv(Wd[:, idx], nbrs, cfg, episode_len=DAY, forecast=fc[:, idx], seed=0)
            for name, pol in policies().items():
                if name in ("equal_split", "expert_current", "expert_oracle") and fname == "yesterday":
                    continue
                key = name if name != "expert_forecast" else f"expert_forecast_{fname}"
                per[key] = [run_policy(env, pol, 144 * k + 1, DAY)["mean_reward"] for k in range(7)]
        summ = {k: dict(zip(("mean", "ci95"), map(lambda x: round(x, 4), ci95(v))), per_day=[round(x, 4) for x in v]) for k, v in per.items()}
        gain = np.array(per["expert_forecast_v2"]) - np.array(per["expert_current"])
        gap = np.array(per["expert_oracle"]) - np.array(per["expert_current"])
        summ["paired_gain_v2_over_current"] = dict(zip(("mean", "ci95"), map(lambda x: round(x, 4), ci95(gain))), days_positive=int((gain > 0).sum()))
        summ["gap_recovered"] = round(float(gain.sum() / gap.sum()), 4)
        B[str(n)] = summ
        print("B", n, {k: v["mean"] for k, v in summ.items() if isinstance(v, dict) and "mean" in v}, "gap recovered", summ["gap_recovered"], flush=True)
    report["heldout_week"] = {"days": "16-22 Dec 2013 (days 45-51)", "results": B}

    # ---- C. MARL fed the v2 forecast ----------------------------------------------------
    C = []
    for seed in ex.SEEDS:
        m = ex.run_agent(SharedMADDPG, ex.make_env(demand, cells_full, 5, f, seed=seed), seed, 300)
        C.append(round(m["mean_reward"], 4))
        print("C marl v2-forecast seed", seed, C[-1], flush=True)
    old = json.loads((ROOT / "reports" / "unified" / "results.json").read_text(encoding="utf-8"))["ablation_marl_n25"]
    ref = {k: [round(x["mean_reward"], 4) for x in v] for k, v in old.items()}
    ref["tgnn_v2"] = C
    report["marl_ablation_n25"] = ref
    report["seconds"] = round(time.time() - t_start)
    (ROOT / "reports" / "unified" / "forecast_value_v2.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote forecast_value_v2.json")


if __name__ == "__main__":
    main()
