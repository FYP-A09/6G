"""Write the unified slicing dataset to disk.

One row per (10-minute interval, cell): real traffic and topology from Milan,
per-slice demand, TGNN forecast, an allocation action, and the resulting QoS,
SLA outcome and reward. Actions come from a logged behaviour policy (the expert
headroom policy plus Dirichlet exploration noise), so the file can also be used
for offline RL; the environment in env.py regenerates outcomes for any other
policy.

Outputs:
  data/processed/unified/unified_dataset.csv.gz   full 7 days x 100 cells (gitignored)
  reports/unified/unified_dataset_sample.csv       day 1, central 5x5 block (committed)
  reports/unified/dataset_checks.json              realism checks against the reference telemetry
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from unified.data import GRID, block_cells, grid_neighbours, load_block_activity  # noqa: E402
from unified.env import CALIBRATION, SLICES, ModelConfig, UnifiedSlicingEnv, demand_scale, headroom  # noqa: E402

MILAN = r"E:\FYP DATA\6G\data\raw\milan_telecom_italia_full\data"
ROOT = Path(__file__).resolve().parents[2]
TRAIN_END = 720


def generate(seed: int = 0, exploration: float = 0.3) -> pd.DataFrame:
    data = load_block_activity(MILAN)
    cells = [int(c) for c in data["cells"]]
    cfg = ModelConfig(latency_sigma=CALIBRATION["latency_sigma"])
    demand = data["activity"] * demand_scale(data["activity"][:TRAIN_END], cfg)
    forecast = np.load(ROOT / "data" / "processed" / "unified" / "tgnn_forecast.npz")["forecast"]
    nb = grid_neighbours(cells)
    env = UnifiedSlicingEnv(demand, nb, cfg, forecast=forecast, seed=seed)
    rng = np.random.default_rng(seed)
    env.reset(start=0)
    env.t_end = env.T - 1
    rows, done = [], False
    while not done:
        t = env.t
        base = headroom(env, None, "current")
        g = rng.gamma(np.maximum(base * 20, 0.05) + 0.05)  # per-cell Dirichlet centred on the expert action
        noisy = g / g.sum(axis=1, keepdims=True)
        action = (1 - exploration) * base + exploration * noisy
        neighbour_load = env.nbr_matrix @ env.load
        _, reward, team, done, info = env.step(action)
        ts = pd.to_datetime(int(data["timestamps"][t + 1]), unit="ms", utc=True).tz_convert("Europe/Rome")
        for i, cid in enumerate(cells):
            r, c = divmod(cid - 1, GRID)
            row = {"interval": t + 1, "time_local": ts.isoformat(), "split": "train" if t + 1 < TRAIN_END else "test",
                   "cell_id": cid, "grid_row": r, "grid_col": c,
                   "neighbours": ";".join(str(cells[j]) for j in nb[i]),
                   "neighbour_load": round(float(neighbour_load[i]), 4),
                   "capacity_mbps": round(float(info["capacity_mbps"][i]), 3),
                   "idle_share": round(float(env.prev_action[i, 3]), 4)}
            for s, name in enumerate(SLICES):
                row.update({
                    f"{name}_demand_mbps": round(float(info["demand_mbps"][i, s]), 4),
                    f"{name}_forecast_mbps": round(float(forecast[t, i, s]), 4),
                    f"{name}_share": round(float(env.prev_action[i, s]), 4),
                    f"{name}_allocated_mbps": round(float(info["allocated_mbps"][i, s]), 4),
                    f"{name}_utilisation": round(float(info["utilisation"][i, s]), 4),
                    f"{name}_served_mbps": round(float(info["served"][i, s]), 4),
                    f"{name}_latency_p50_ms": round(float(info["latency_p50_ms"][i, s]), 3),
                    f"{name}_latency_p95_ms": round(float(info["latency_p95_ms"][i, s]), 3),
                    f"{name}_jitter_ms": round(float(info["jitter_ms"][i, s]), 3),
                    f"{name}_loss": float(f"{info['loss'][i, s]:.6g}"),
                    f"{name}_sla_ok": int(info["sla_ok"][i, s]),
                })
            row.update({"reward": round(float(reward[i]), 5), "team_reward": round(float(team[i]), 5)})
            rows.append(row)
    return pd.DataFrame(rows)


def realism_checks(df: pd.DataFrame) -> dict:
    """Compare the generated eMBB SLA rate against the reference dataset's own SLA label,
    both as a function of utilisation (the reference label was not used to build our model)."""
    ref_u = np.array(CALIBRATION["utilisation"])
    ref_ok = np.array(CALIBRATION["reference_sla_ok_rate"])
    edges = np.arange(0.25, 1.0001, 0.05)
    u, ok = df["eMBB_utilisation"].to_numpy(), df["eMBB_sla_ok"].to_numpy()
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (u >= lo) & (u < hi)
        if m.sum() < 50:
            continue
        centre = (lo + hi) / 2
        rows.append({"utilisation": round(centre, 3), "generated_sla_ok": round(float(ok[m].mean()), 3),
                     "reference_sla_ok": round(float(np.interp(centre, ref_u, ref_ok)), 3), "n": int(m.sum())})
    hourly = df.assign(hour=pd.to_datetime(df["time_local"]).dt.hour).groupby("hour")["eMBB_demand_mbps"].mean()
    return {"embb_sla_vs_utilisation": rows,
            "mean_abs_gap": round(float(np.mean([abs(r["generated_sla_ok"] - r["reference_sla_ok"]) for r in rows])), 3),
            "hourly_mean_embb_demand_mbps": hourly.round(2).tolist(),
            "rows": int(len(df)), "cells": int(df["cell_id"].nunique()), "intervals": int(df["interval"].nunique()),
            "sla_ok_rate": {s: round(float(df[f"{s}_sla_ok"].mean()), 4) for s in SLICES}}


if __name__ == "__main__":
    df = generate()
    full = ROOT / "data" / "processed" / "unified" / "unified_dataset.csv.gz"
    df.to_csv(full, index=False, compression="gzip")
    sample_cells = set(block_cells(5))
    sample = df[(df["interval"] < 145) & df["cell_id"].isin(sample_cells)]
    out = ROOT / "reports" / "unified"
    out.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out / "unified_dataset_sample.csv", index=False)
    checks = realism_checks(df)
    (out / "dataset_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(f"wrote {full} ({len(df):,} rows, {len(df.columns)} columns); sample {len(sample):,} rows")
    print(json.dumps({k: v for k, v in checks.items() if k != "hourly_mean_embb_demand_mbps"}, indent=1))
