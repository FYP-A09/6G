"""City-scale data for the Slice Twin monitor: the same held-out day on all 10,000 Milan cells.

Same day (Wed 6 Nov 2013), same demand scale and the same three controllers as
showcase.py, run on the whole grid. The TGNN forecaster trained on the 100-cell
block is applied to the 10,000-cell graph unchanged (GraphSAGE is inductive), which
also tests whether it generalises beyond the cells it was trained on.

Scripted events (each labelled in the output; none of them is a real incident):
  the six block events of showcase.py, at the same cells and times, plus
  03:00-03:40  mMTC storm in the north-west (a firmware push to industrial meters)
  11:00-13:00  wide-area interference in the east
  16:00-18:00  three adjacent cells out in the south (storm damage)
  19:30-23:00  stadium crowd around San Siro

Output (for the app): city.json (meta, events, landmarks, city time series, summary)
and city.bin.gz: per interval and cell, a log-scaled demand byte, then per controller
a load byte (x250) and a flags byte (bit 0-2 slice SLA met, bit 3 event, bit 4 outage).
Also reports/unified/showcase_city_summary.json.

Run: python src/unified/showcase_city.py <out_dir>
"""
from __future__ import annotations

import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from unified.forecast import WINDOW  # noqa: E402
from unified import forecast_city_v2 as v2  # noqa: E402
from unified.data import GRID, block_cells, grid_neighbours, load_block_activity  # noqa: E402
from unified.env import SLICES, ModelConfig, UnifiedSlicingEnv, demand_scale, equal_split, headroom  # noqa: E402
from unified.showcase import DAY_START, EVENTS as BLOCK_EVENTS, MILAN, PER_DAY, TRAIN_END, iod  # noqa: E402
from unified.twin_dataset import BASE, apply_events, full_grid  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CODES = {"flash_crowd": 1, "cell_outage": 2, "interference_spike": 3, "urllc_burst": 4, "mmtc_storm": 5, "capacity_degradation": 6}
LANDMARKS = {"Duomo": (45.4641, 9.1919), "Milano Centrale": (45.4864, 9.2045), "San Siro": (45.4781, 9.1240),
             "Linate airport": (45.4451, 9.2767), "Navigli": (45.4505, 9.1730), "Città Studi": (45.4780, 9.2270)}
B0 = 50 - 5  # the block's first row and column on the grid


def all_centroids() -> np.ndarray:
    gj = json.loads(Path(MILAN, "milano-grid.geojson").read_text(encoding="utf-8"))
    out = np.zeros((GRID * GRID, 2))
    for f in gj["features"]:
        ring = np.array(f["geometry"]["coordinates"][0][:-1])
        out[f["properties"]["cellId"] - 1] = (ring[:, 1].mean(), ring[:, 0].mean())
    return out


def nearest(cent: np.ndarray, lat: float, lon: float) -> int:
    return int(np.argmin((cent[:, 0] - lat) ** 2 + ((cent[:, 1] - lon) * np.cos(np.radians(lat))) ** 2))


def disc(centre: int, radius: int) -> list[int]:
    r0, c0 = divmod(centre, GRID)
    return [r * GRID + c for r in range(max(0, r0 - radius), min(GRID, r0 + radius + 1))
            for c in range(max(0, c0 - radius), min(GRID, c0 + radius + 1)) if abs(r - r0) + abs(c - c0) <= radius]


def city_events(cent: np.ndarray) -> list[dict]:
    ev = []
    for etype, label, t0, t1, cells, params, desc in BLOCK_EVENTS:
        glob = [(B0 + c // 10) * GRID + B0 + c % 10 for c in cells]
        ev.append(dict(type=etype, label=label, t0=t0, t1=t1, cells=glob, params=params, description=desc, scope="block"))
    nw, east, south = nearest(cent, 45.515, 9.085), nearest(cent, 45.470, 9.255), nearest(cent, 45.405, 9.190)
    ev += [
        dict(type="mmtc_storm", label="Meter firmware push", t0=(3, 0), t1=(3, 40), cells=disc(nw, 4), params={"mmtc_x": 10.0},
             description="A firmware update reaches every industrial meter in the north-west at once: mMTC x10.", scope="city"),
        dict(type="interference_spike", label="Wide-area interference", t0=(11, 0), t1=(13, 0), cells=disc(east, 3),
             params={"coupling_x": 3.0}, description="External interference across the east triples inter-cell coupling.", scope="city"),
        dict(type="flash_crowd", label="Stadium crowd, San Siro", t0=(19, 30), t1=(23, 0),
             cells=disc(nearest(cent, *LANDMARKS["San Siro"]), 2),
             params={"embb_x": 4.0, "urllc_x": 2.0, "mmtc_x": 1.2},
             description="A scripted evening event at the stadium: eMBB x4, URLLC x2 around San Siro.", scope="city"),
    ]
    for k, c in enumerate((south, south + 1, south + GRID)):
        ev.append(dict(type="cell_outage", label="Storm damage, south" if k == 0 else f"Storm damage, south ({k + 1})",
                       t0=(16, 0), t1=(18, 0), cells=[c], params={"capacity_x": 0.0, "rehome_fraction": 0.7},
                       description="Three adjacent cells lose power; 70% of their users re-attach to neighbours.", scope="city"))
    return ev


def run(env, policy, lo: int):
    env.reset(start=lo)
    env.t_end = lo + PER_DAY
    out = {"load": [], "ok": [], "reward": [], "served": [], "alloc": [], "share": []}
    done = False
    while not done:
        _, r, _, done, info = env.step(policy(env))
        out["load"].append(env.load.astype(np.float32)); out["ok"].append(info["sla_ok"]); out["reward"].append(r.astype(np.float32))
        out["served"].append(info["served"].sum(axis=0)); out["alloc"].append(info["allocated_mbps"].sum(axis=0))
    return {k: np.array(v) for k, v in out.items() if v}


def main(out_dir: str) -> None:
    t_start = time.time()
    cfg = ModelConfig()
    block = load_block_activity(MILAN)
    scale = demand_scale(block["activity"][:TRAIN_END], cfg)
    grid = full_grid(BASE / "milan_full_grid_62d.npy")
    lo = DAY_START - WINDOW  # history the forecaster needs before the day starts
    act = np.array(grid[lo:DAY_START + PER_DAY], dtype=np.float32)
    bidx = [c - 1 for c in block_cells(10)]
    assert np.allclose(act[WINDOW:, bidx], block["activity"][DAY_START:DAY_START + PER_DAY], atol=1e-3), "grid cache != block cache"
    demand = (act * scale).astype(np.float32)
    nbrs = grid_neighbours(list(range(1, GRID * GRID + 1)))
    cent = all_centroids()
    events = city_events(cent)
    rows = [{"event_id": i, "type": e["type"], "code": CODES[e["type"]], "start_interval": WINDOW + iod(*e["t0"]),
             "end_interval": WINDOW + iod(*e["t1"]), "cells_local": e["cells"], "params": e["params"]} for i, e in enumerate(events)]
    cap_f, coup_f, code, _, r_in, r_out = apply_events(demand, pd.DataFrame(rows), nbrs)

    ck = torch.load(v2.MODEL, weights_only=False)
    model = v2.CityTGNN(); model.load_state_dict(ck["state"]); cell_scale = ck["cell_scale"]
    t0 = time.time()
    t_hist = DAY_START - WINDOW - v2.DAY - WINDOW                 # v2 also looks one day back
    Dv = v2.load_range(grid, scale, t_hist, DAY_START + PER_DAY)
    Dv[lo - t_hist:] = demand                                      # the forecaster sees the incidents' effect on today's traffic, as a live one would
    ts = np.arange(DAY_START - WINDOW, DAY_START + PER_DAY - 1)
    fc = demand.astype(np.float64).copy()
    fc[:len(ts)] = v2.predict(model, Dv, t_hist, ts, cell_scale)
    day = slice(WINDOW, WINDOW + PER_DAY)
    truth, persist = demand[WINDOW:WINDOW + PER_DAY].astype(np.float64), demand[WINDOW - 1:WINDOW + PER_DAY - 1].astype(np.float64)
    fmse = ((fc[WINDOW - 1:WINDOW + PER_DAY - 1] - truth) ** 2).mean(axis=(0, 1))
    pmse = ((persist - truth) ** 2).mean(axis=(0, 1))
    print(f"city TGNN on 10,000 cells: {time.time() - t0:.0f}s, mse {fmse.round(3)} vs persistence {pmse.round(3)}", flush=True)
    fr = json.loads((ROOT / "reports" / "unified" / "forecast_city_v2.json").read_text(encoding="utf-8"))
    better = fr["regions"]["city_10000"]["tgnn_v2"]["better_than_persistence"]
    old = json.loads((ROOT / "reports" / "unified" / "forecast_city.json").read_text(encoding="utf-8"))
    naive = old["block_forecaster_on_city"]["outer_9600"]["better_than_persistence"][0]
    note = (f"Yes, once trained properly. The predictive controller uses a TGNN trained on random patches of the whole grid, "
            f"with each cell's traffic scaled by its own level, on 39 days of real traffic. On a held-out week (16-22 Dec) it beats "
            f"'repeat the last value' by {better[0]:.0%} (eMBB), {better[1]:.0%} (URLLC) and {better[2]:.0%} (mMTC) across all 10,000 cells, "
            f"and it never saw this demo day. A first forecaster trained only on the 100 block cells did not transfer: on the quiet "
            f"outer cells its error was {1 - naive:.0f} times that baseline's error, because it had learned the centre's traffic level.")

    def make_env(forecast=None):
        return UnifiedSlicingEnv(demand, nbrs, cfg, episode_len=PER_DAY, forecast=forecast,
                                 capacity_factor=cap_f, coupling_factor=coup_f)

    controllers = {"static": (make_env(), lambda e: equal_split(e, None)),
                   "reactive": (make_env(), lambda e: headroom(e, None, "current")),
                   "predictive": (make_env(fc.astype(np.float32)), lambda e: headroom(e, None, "forecast"))}
    d_day, code_day = demand[day], code[day]
    total = d_day.sum(axis=2)
    dmax = float(total.max())
    blob = [np.rint(255 * np.log1p(total) / np.log1p(dmax)).astype(np.uint8).ravel()]
    in_block = np.zeros(GRID * GRID, bool); in_block[bidx] = True
    series = {"demand_gbps": (d_day.sum(axis=1) / 1000).round(3).T.tolist()}
    summary = {"forecast_mse_city": {"tgnn": fmse.round(4).tolist(), "persistence": pmse.round(4).tolist(),
                                     "tgnn_better_by": (1 - fmse / pmse).round(4).tolist()}}
    outage = (cap_f[day] == 0)
    for key, (env, pol) in controllers.items():
        t0 = time.time()
        rec = run(env, pol, WINDOW - 1)
        ok = rec["ok"]
        flags = (ok[:, :, 0] | (ok[:, :, 1] << 1) | (ok[:, :, 2] << 2) | ((code_day > 0) << 3) | (outage << 4)).astype(np.uint8)
        blob += [np.rint(np.clip(rec["load"], 0, 1) * 250).astype(np.uint8).ravel(), flags.ravel()]
        ev = code_day > 0
        summary[key] = {"sla_ok": dict(zip(SLICES, ok.mean(axis=(0, 1)).round(4).tolist())),
                        "sla_ok_during_events": dict(zip(SLICES, ok[ev].mean(axis=0).round(4).tolist())),
                        "cell_intervals_all_ok": round(float(ok.all(axis=2).mean()), 4),
                        "mean_reward": round(float(rec["reward"].mean()), 4),
                        "served_fraction": round(float(rec["served"].sum() / d_day.sum()), 4),
                        "block_mean_reward": round(float(rec["reward"][:, in_block].mean()), 4),
                        "outer_mean_reward": round(float(rec["reward"][:, ~in_block].mean()), 4)}
        series[key] = {"all_ok": ok.all(axis=2).mean(axis=1).round(4).tolist(),
                       "slice_ok": ok.mean(axis=1).round(4).T.tolist(),
                       "alloc_gbps": (rec["alloc"] / 1000).round(3).T.tolist(),
                       "mean_load": rec["load"].mean(axis=1).round(4).tolist()}
        print(f"{key}: {json.dumps(summary[key])} ({time.time() - t0:.0f}s)", flush=True)

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "city.bin.gz").write_bytes(gzip.compress(np.concatenate(blob).tobytes(), 6))
    marks = []
    for name, (la, lo_) in LANDMARKS.items():
        i = nearest(cent, la, lo_)
        marks.append({"name": name, "r": i // GRID, "c": i % GRID})
    payload = {"meta": {"grid": GRID, "cells": GRID * GRID, "intervals": PER_DAY, "block": {"r0": B0, "c0": B0, "size": 10},
                        "demand_max_mbps": round(dmax, 2), "controllers": list(controllers),
                        "layout": "demand u8 [t, cell], then per controller load u8 (x250) and flags u8 (bit0-2 SLA, bit3 event, bit4 outage)",
                        "row0": "south", "forecaster_note": note, "source": "Real Telecom Italia Milan traffic on all 10,000 grid cells, Wed 6 Nov 2013 (held out), "
                        "scripted events, calibrated QoS model; TGNN trained on the central 100 cells only."},
               "events": [{"id": i, "type": e["type"], "label": e["label"], "start": iod(*e["t0"]), "end": iod(*e["t1"]),
                           "cells": e["cells"], "description": e["description"], "scope": e["scope"]} for i, e in enumerate(events)],
               "landmarks": marks, "series": series, "summary": summary}
    (out / "city.json").write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    (ROOT / "reports" / "unified" / "showcase_city_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out} ({(out / 'city.bin.gz').stat().st_size / 1e6:.2f} MB bin, "
          f"{(out / 'city.json').stat().st_size / 1e6:.2f} MB json) in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1])
