"""Does the unified environment's latency-against-load curve look like a real 5G NR cell's?

Runs Simu5G 1.7.0 (WSL, opp_env) on one NR gNodeB serving 10 UEs with downlink UDP traffic (src/digital_twin/simu5g_scenario/),
sweeping the offered load, and records the end-to-end packet delay (p50, p95, p99) and the received throughput. The cell's capacity
is the highest received throughput seen. Utilisation = offered load / capacity, the same quantity the unified environment's
calibrated latency curve is a function of (qos_calibration.json, fitted on the 6G RAN telemetry dataset).

The two latencies have different absolute scales (a telemetry dataset's per-window latency against one simulator's per-packet delay),
so the comparison is of shape: delays normalised by their value at low load, the utilisation at which p95 doubles ("knee"), and the
rank correlation of the two curves over the common range.

Run: python src/unified/simu5g_sweep.py -> reports/unified/simu5g_sweep.json, docs/figures/simu5g_latency_vs_load.png
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = "/mnt/c/Users/keert/Downloads/fy/6G/src/digital_twin/simu5g_scenario/run_point.sh"
OFFERED = (6, 12, 20, 30, 38, 44, 48, 52, 56, 60, 70)      # total Mbps over 10 UEs
REPS, SIM = (0, 1, 2), 6
UES, MSG = 10, 1000


def run(rate, rep):
    interval = MSG * 8 / (rate * 1e6 / UES)
    cmd = f"cd ~/6g-sim-workspace && timeout 900 opp_env run simu5g-1.7.0 -c 'bash {SCRIPT} {interval:.8f} {SIM} {rep} {rate}' 2>&1 | tail -1"
    out = subprocess.run(["wsl", "-e", "bash", "-lc", cmd], capture_output=True, text=True, timeout=1000).stdout.strip().splitlines()
    return json.loads(out[-1])


def main():
    points = []
    for rate in OFFERED:
        for rep in REPS:
            r = run(rate, rep)
            r["offered_mbps"] = rate
            points.append(r)
            print(r, flush=True)
    capacity = max(p["received_mbps"] for p in points)
    by = {}
    for p in points:
        by.setdefault(p["offered_mbps"], []).append(p)
    rows = []
    for rate in OFFERED:
        ps = by[rate]
        rows.append({"offered_mbps": rate, "utilisation": round(rate / capacity, 3),
                     **{k: round(float(np.median([p[k] for p in ps])), 3) for k in ("p50_ms", "p95_ms", "p99_ms", "received_mbps")}})
    cal = json.loads((ROOT / "src" / "unified" / "qos_calibration.json").read_text(encoding="utf-8"))
    cu, c50, c95 = np.array(cal["utilisation"]), np.array(cal["lat_p50_ms"]), np.array(cal["lat_p95_ms"])
    u = np.array([r["utilisation"] for r in rows])
    s50, s95 = np.array([r["p50_ms"] for r in rows]), np.array([r["p95_ms"] for r in rows])
    ok = u <= 1.0
    low = u[ok].min()
    sim95n, sur95n = s95[ok] / s95[ok][0], np.interp(u[ok], cu, c95) / np.interp(u[ok][0], cu, c95)
    knee = lambda x, y: float(x[np.argmax(y >= 2 * y[0])]) if (y >= 2 * y[0]).any() else None
    from scipy.stats import spearmanr
    rho = spearmanr(sim95n, sur95n)
    out = {"capacity_mbps": capacity, "ues": UES, "message_bytes": MSG, "sim_seconds": SIM, "repetitions": len(REPS), "rows": rows,
           "knee_p95_doubles_at_utilisation": {"simu5g": knee(u[ok], sim95n), "surrogate": knee(u[ok], sur95n)},
           "spearman_rank_correlation_p95_normalised": round(float(rho.correlation), 3),
           "normalised_p95": {"utilisation": [round(float(x), 3) for x in u[ok]], "simu5g": [round(float(x), 2) for x in sim95n], "surrogate": [round(float(x), 2) for x in sur95n]}}
    (ROOT / "reports" / "unified" / "simu5g_sweep.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.plot(u[ok], sim95n, "o-", color="#2a78d6", linewidth=2, label="Simu5G NR cell, p95 delay")
    ax.plot(u[ok], sur95n, "s--", color="#eb6834", linewidth=2, label="Unified environment, calibrated p95 latency")
    ax.set_yscale("log"); ax.set_xlabel("Utilisation (offered load / cell capacity)", color="#52514e")
    ax.set_ylabel("p95 latency, relative to the lowest load", color="#52514e")
    ax.set_title("Latency against load: simulator and surrogate", loc="left", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(color="#e1e0d9"); ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(ROOT / "docs" / "figures" / "simu5g_latency_vs_load.png", dpi=200)


if __name__ == "__main__":
    main()
