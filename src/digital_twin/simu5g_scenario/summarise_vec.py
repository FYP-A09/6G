"""Summarise a Simu5G .vec file from the load sweep: end-to-end delay percentiles and received throughput.

Run inside WSL (system python3): python3 summarise_vec.py <file.vec> <offered_mbps> <message_bytes> <sim_seconds> <warmup_seconds>
Prints one JSON line.
"""
import json
import sys

import numpy as np

path, offered, msg, sim, warm = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
ids = {}
delays, times = [], []
with open(path) as f:
    for line in f:
        if line.startswith("vector "):
            parts = line.split()
            if parts[3].startswith("endToEndDelay:vector") and ".ue[" in parts[2]:
                ids[parts[1]] = parts[2]
        elif line and line[0].isdigit():
            p = line.split()
            if p[0] in ids and len(p) >= 4:
                t, v = float(p[2]), float(p[3])
                if t >= warm:
                    delays.append(v); times.append(t)
d = np.array(delays) * 1000.0
received_mbps = len(d) * msg * 8 / 1e6 / max(sim - warm, 1e-9)
out = {"offered_mbps": offered, "received_mbps": round(received_mbps, 2), "packets": int(len(d))}
if len(d):
    out.update({"p50_ms": round(float(np.percentile(d, 50)), 3), "p95_ms": round(float(np.percentile(d, 95)), 3), "p99_ms": round(float(np.percentile(d, 99)), 3),
                "mean_ms": round(float(d.mean()), 3)})
print(json.dumps(out))
