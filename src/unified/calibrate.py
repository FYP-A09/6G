"""Fit the QoS model's latency curves from a published RAN telemetry dataset.

Source: Kaggle "Wireless 6G network dataset of resource allocation"
(nizamuddinmaitlo, CC0), 30,000 client-windows with PRB utilisation and
p50/p95 latency. Note that this dataset is itself simulated by its authors; it
is used here as an independent reference so our latency-vs-load behaviour is
not invented. Output: qos_calibration.json (committed), so the environment does
not need the raw file at runtime.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "qos_calibration.json"
DEFAULT_SRC = r"E:\FYP DATA\6G\data\raw\6g_ran_telemetry_fl\6g_fl_telemetry_200_clients.csv"


def fit(path: str = DEFAULT_SRC, width: float = 0.05, min_count: int = 30) -> dict:
    df = pd.read_csv(path, usecols=["PRBmean", "lat_p50_ms", "lat_p95_ms", "sla_ok_current"])
    u = (df["PRBmean"] / 100.0).clip(0, 1)
    edges = np.arange(0, 1.0 + 1e-9, width)
    centres, p50, p95, ok, counts = [], [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (u >= lo) & (u < hi if hi < 1 else u <= hi)
        if m.sum() < min_count:
            continue
        centres.append(float(u[m].mean()))
        p50.append(float(df.loc[m, "lat_p50_ms"].mean()))
        p95.append(float(df.loc[m, "lat_p95_ms"].mean()))
        ok.append(float(df.loc[m, "sla_ok_current"].mean()))
        counts.append(int(m.sum()))
    # Latency must not fall as load rises; small sampling dips are removed.
    p50 = np.maximum.accumulate(p50).tolist()
    p95 = np.maximum.accumulate(np.maximum(p95, p50)).tolist()
    sigma = fit_latency_sigma(np.array(p95), np.array(ok))
    return {"source": Path(path).name, "rows": int(len(df)), "utilisation": centres,
            "lat_p50_ms": p50, "lat_p95_ms": p95, "reference_sla_ok_rate": ok, "counts": counts,
            "latency_sigma": sigma}


def fit_latency_sigma(p95: np.ndarray, ok: np.ndarray, target_ms: float = 50.0) -> float:
    """Spread of a per-window lognormal latency factor such that P(p95 * eps <= target)
    matches the reference SLA rate at each utilisation (least squares over bins).
    Without it the model's SLA outcome is a hard step at one utilisation."""
    from math import erf, log, sqrt
    best, best_err = 0.0, float("inf")
    for s in np.arange(0.02, 1.0, 0.005):
        pred = np.array([0.5 * (1 + erf(log(target_ms / v) / (s * sqrt(2)))) for v in p95])
        err = float(((pred - ok) ** 2).sum())
        if err < best_err:
            best, best_err = float(s), err
    return round(best, 3)


if __name__ == "__main__":
    result = fit(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SRC)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for row in zip(result["utilisation"], result["lat_p50_ms"], result["lat_p95_ms"], result["reference_sla_ok_rate"], result["counts"]):
        print("u=%.3f  p50=%5.1f  p95=%5.1f  ref_sla_ok=%.2f  n=%d" % row)
    print("wrote", OUT)
