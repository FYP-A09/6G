"""An alternative latency-against-load curve measured in Simu5G, for sensitivity analysis.

The unified environment's default curve (qos_calibration.json) rises gradually from about 50% utilisation, as in the 6G RAN
telemetry it was fitted to. A packet-level NR cell simulated in Simu5G (simu5g_sweep.py) keeps its delay almost flat until the
cell saturates and then collapses. apply() replaces the calibration arrays, in place, by a curve with that shape: the low-load
latency stays at the calibrated value, the growth factors are the measured ones (p95 about 1.4x between 60% and 95% of capacity,
about 88x at saturation; p50 about 1.3x), where "capacity" is the throughput at which the simulated cell stops carrying more.

Call apply() before any environment is built, or set the environment variable QOS=simu5g for scripts that honour it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def apply() -> dict:
    from unified import env as envmod
    sweep = json.loads((ROOT / "reports" / "unified" / "simu5g_sweep.json").read_text(encoding="utf-8"))
    rows = sweep["rows"]
    plateau = float(np.mean([r["received_mbps"] for r in rows if r["offered_mbps"] >= 52 and r["offered_mbps"] <= 60]))
    u = np.array([r["offered_mbps"] / plateau for r in rows])
    p95 = np.array([r["p95_ms"] for r in rows]); p50 = np.array([r["p50_ms"] for r in rows])
    g95, g50 = p95 / p95[0], p50 / p50[0]
    grid = np.array([0.1, 0.5, 0.6, 0.7, 0.85, 0.95, 0.99, 1.0])
    f95 = np.array([1.0, 1.0, np.interp(0.6, u, g95), np.interp(0.7, u, g95), np.interp(0.85, u, g95), np.interp(0.95, u, g95), np.interp(0.95, u, g95), float(g95[np.argmin(abs(u - 1.03))])])
    f50 = np.array([1.0, 1.0, np.interp(0.6, u, g50), np.interp(0.7, u, g50), np.interp(0.85, u, g50), np.interp(0.95, u, g50), np.interp(0.95, u, g50), float(g50[np.argmin(abs(u - 1.03))])])
    cal = envmod.CALIBRATION
    base95, base50 = cal["lat_p95_ms"][0], cal["lat_p50_ms"][0]
    cal["utilisation"] = grid.tolist()
    cal["lat_p95_ms"] = (base95 * f95).tolist()
    cal["lat_p50_ms"] = (base50 * np.minimum(f50, f95)).tolist()
    return {"capacity_mbps": plateau, "p95_ms": [round(x, 1) for x in cal["lat_p95_ms"]], "utilisation": grid.tolist()}


def maybe_apply() -> bool:
    import os
    if os.environ.get("QOS", "") == "simu5g":
        print("QoS curve: Simu5G-shaped", apply(), flush=True)
        return True
    return False
