"""Data for the Slice Twin monitor: one held-out real day, scripted events, three controllers.

Day: Wed 6 Nov 2013 (intervals 720-863, outside the training days) on the central
10 x 10 Milan block. Real traffic, with these scripted events injected so the
controllers have something to react to (each is labelled in the output):

  07:00-07:30  mMTC storm (synchronised smart-meter reporting)
  09:30-11:00  interference spike
  13:00-14:20  cell outage, 70% of its traffic re-homes to neighbours
  15:30-17:30  capacity degradation (backhaul fault)
  18:30-21:00  flash crowd (evening event)
  20:10-20:50  URLLC burst (emergency services)

Controllers, all on the same environment (env.py, noise-free QoS as in the experiments):
  static      equal split
  reactive    expert headroom policy on the demand it sees now
  predictive  the same expert fed the TGNN forecast of the next interval
  oracle      the same expert with the true next demand (summary only, a ceiling)

Output: a compact JSON consumed by the app (path given on the command line), and
reports/unified/showcase_summary.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from unified.forecast import predict, train_forecaster  # noqa: E402  (also puts src/tgnn on the path)
from model import TGNNPredictor  # noqa: E402
from unified.data import GRID, grid_neighbours, load_block_activity  # noqa: E402
from unified.env import SLICES, ModelConfig, UnifiedSlicingEnv, demand_scale, equal_split, headroom  # noqa: E402
from unified.twin_dataset import apply_events  # noqa: E402

MILAN = r"E:\FYP DATA\6G\data\raw\milan_telecom_italia_full\data"
ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "data" / "processed" / "unified" / "tgnn_forecaster.pt"
TRAIN_END, DAY_START, PER_DAY = 720, 720, 144


def iod(hh: int, mm: int) -> int:
    return hh * 6 + mm // 10


def local(r: int, c: int) -> int:
    return r * 10 + c


def ring(r: int, c: int, radius: int) -> list[int]:
    return [local(i, j) for i in range(10) for j in range(10) if abs(i - r) + abs(j - c) <= radius]


EVENTS = [
    ("mmtc_storm", "mMTC storm", (7, 0), (7, 30), ring(2, 7, 2), {"mmtc_x": 8.0},
     "Thousands of smart meters report at once: mMTC demand x8 for 30 minutes."),
    ("interference_spike", "Interference spike", (9, 30), (11, 0), ring(6, 2, 1), {"coupling_x": 3.0},
     "External interference triples inter-cell coupling, so each cell's usable capacity drops."),
    ("cell_outage", "Cell outage", (13, 0), (14, 20), [local(4, 4)], {"capacity_x": 0.0, "rehome_fraction": 0.7},
     "A cell goes dark. 70% of its users re-attach to the four neighbours; the rest lose service."),
    ("capacity_degradation", "Backhaul degradation", (15, 30), (17, 30), [local(1, 1)], {"capacity_x": 0.5},
     "A backhaul fault halves one cell's capacity for two hours."),
    ("flash_crowd", "Flash crowd", (18, 30), (21, 0), ring(5, 6, 2), {"embb_x": 3.0, "urllc_x": 1.8, "mmtc_x": 1.2},
     "An evening event draws a crowd: eMBB x3, URLLC x1.8 across 13 cells."),
    ("urllc_burst", "URLLC burst", (20, 10), (20, 50), [local(7, 3), local(7, 4)], {"urllc_x": 5.0},
     "Emergency-services traffic: URLLC x5 on two cells."),
]
CODES = {"flash_crowd": 1, "cell_outage": 2, "interference_spike": 3, "urllc_burst": 4, "mmtc_storm": 5, "capacity_degradation": 6}


def centroids(cells: list[int]) -> dict[int, tuple[float, float]]:
    gj = json.loads(Path(MILAN, "milano-grid.geojson").read_text(encoding="utf-8"))
    out = {}
    wanted = set(cells)
    for f in gj["features"]:
        cid = f["properties"]["cellId"]
        if cid in wanted:
            ring_ = np.array(f["geometry"]["coordinates"][0][:-1])
            out[cid] = (float(ring_[:, 1].mean()), float(ring_[:, 0].mean()))
    return out


def forecaster(demand, nbrs):
    if MODEL.exists():
        ck = torch.load(MODEL, weights_only=False)
        model = TGNNPredictor(embedding_dim=5)
        model.load_state_dict(ck["state"])
        return model, ck["scale"]
    out = train_forecaster(demand, nbrs, TRAIN_END, epochs=30)
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": out["model"].state_dict(), "scale": out["scale"]}, MODEL)
    return out["model"], out["scale"]


def run(env, policy, record: bool):
    env.reset(start=DAY_START - 1)
    env.t_end = DAY_START - 1 + PER_DAY
    keys = ("allocated_mbps", "served", "latency_p95_ms", "sla_ok", "capacity_mbps", "reward", "utilisation")
    rec = {k: [] for k in keys + ("share", "load")}
    done = False
    while not done:
        action = policy(env)
        _, _, _, done, info = env.step(action)
        if record:
            for k in keys:
                rec[k].append(info[k])
            rec["share"].append(env.prev_action.copy())
            rec["load"].append(env.load.copy())
        else:
            rec["sla_ok"].append(info["sla_ok"]); rec["reward"].append(info["reward"]); rec["served"].append(info["served"])
    return {k: np.array(v) for k, v in rec.items() if v}


def summarise(rec, demand_day, code_day) -> dict:
    ok, reward = rec["sla_ok"], rec["reward"]
    ev = code_day > 0
    total = demand_day.sum(axis=2)
    return {"sla_ok": dict(zip(SLICES, ok.mean(axis=(0, 1)).round(4).tolist())),
            "sla_ok_during_events": dict(zip(SLICES, ok[ev].mean(axis=0).round(4).tolist())),
            "mean_reward": round(float(reward.mean()), 4),
            "served_fraction": round(float(rec["served"].sum() / max(total.sum(), 1e-9)), 4),
            "cell_intervals_all_slices_ok": round(float(ok.all(axis=2).mean()), 4)}


def main(out_path: str) -> None:
    data = load_block_activity(MILAN)
    cells = [int(c) for c in data["cells"]]
    nbrs = grid_neighbours(cells)
    cfg = ModelConfig()
    demand = (data["activity"] * demand_scale(data["activity"][:TRAIN_END], cfg)).astype(np.float64)
    model, scale = forecaster(demand, nbrs)

    ev_rows = []
    for i, (etype, label, (h0, m0), (h1, m1), cl, params, desc) in enumerate(EVENTS):
        ev_rows.append({"event_id": i, "type": etype, "code": CODES[etype], "start_interval": DAY_START + iod(h0, m0),
                        "end_interval": DAY_START + iod(h1, m1), "cells_local": cl, "params": params})
    cap_f, coup_f, code, _, r_in, r_out = apply_events(demand, pd.DataFrame(ev_rows), nbrs)
    fc = predict(model, demand, nbrs, scale)

    def make_env(forecast=None):
        return UnifiedSlicingEnv(demand, nbrs, cfg, episode_len=PER_DAY, forecast=forecast,
                                 capacity_factor=cap_f, coupling_factor=coup_f)

    controllers = {
        "static": ("Static", "Equal thirds, never changes", make_env(), lambda e: equal_split(e, None)),
        "reactive": ("Reactive", "Expert rules on the demand it sees now", make_env(), lambda e: headroom(e, None, "current")),
        "predictive": ("Predictive", "Same rules, fed the TGNN forecast of the next 10 minutes", make_env(fc),
                       lambda e: headroom(e, None, "forecast")),
    }
    day = slice(DAY_START, DAY_START + PER_DAY)
    demand_day, code_day = demand[day], code[day]
    policies, summary = {}, {}
    for key, (label, desc, env, pol) in controllers.items():
        rec = run(env, pol, record=True)
        summary[key] = summarise(rec, demand_day, code_day)
        policies[key] = {
            "label": label, "description": desc,
            "share": np.rint(rec["share"] * 1000).astype(int).ravel().tolist(),
            "served": np.round(rec["served"], 2).ravel().tolist(),
            "lat": np.round(np.minimum(rec["latency_p95_ms"], 999), 1).ravel().tolist(),
            "ok": "".join("1" if v else "0" for v in rec["sla_ok"].ravel()),
            "cap": np.round(rec["capacity_mbps"], 1).ravel().tolist(),
            "load": np.rint(rec["load"] * 1000).astype(int).ravel().tolist(),
            "reward": np.rint(rec["reward"] * 1000).astype(int).ravel().tolist(),
        }
    oracle = run(make_env(), lambda e: headroom(e, None, "oracle"), record=False)
    summary["oracle"] = summarise(oracle, demand_day, code_day)

    ll = centroids(cells)
    payload = {
        "meta": {"date": "2013-11-06", "weekday": "Wednesday", "intervals": PER_DAY, "cells": len(cells),
                 "capacity_mbps": cfg.capacity_mbps, "slices": list(SLICES),
                 "target_latency_ms": list(cfg.target_latency_ms), "target_jitter_ms": list(cfg.target_jitter_ms),
                 "source": "Real Telecom Italia Milan traffic (held-out test day) on the central 10x10 grid block, "
                           "scripted events, calibrated QoS model (src/unified)."},
        "cells": [{"id": c, "r": (c - 1) // GRID, "c": (c - 1) % GRID, "lat": round(ll[c][0], 5), "lon": round(ll[c][1], 5),
                   "nb": nbrs[i]} for i, c in enumerate(cells)],
        "times": [f"{(k // 6):02d}:{(k % 6) * 10:02d}" for k in range(PER_DAY)],
        "events": [{"id": i, "type": e[0], "label": e[1], "start": iod(*e[2]), "end": iod(*e[3]), "cells": e[4],
                    "params": e[5], "description": e[6]} for i, e in enumerate(EVENTS)],
        "demand": np.round(demand_day, 2).ravel().tolist(),
        "forecast": np.round(fc[DAY_START - 1:DAY_START - 1 + PER_DAY], 2).ravel().tolist(),
        "code": code_day.astype(int).ravel().tolist(),
        "rehomed": np.round(r_in[day] - r_out[day], 2).ravel().tolist(),
        "policies": policies, "summary": summary,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    (ROOT / "reports" / "unified" / "showcase_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print("wrote", out_path, round(Path(out_path).stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main(sys.argv[1])
