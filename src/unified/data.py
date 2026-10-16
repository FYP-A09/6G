"""Real Milan traffic for a contiguous block of grid cells, as the backbone of the
unified slicing dataset.

Milan's release is the only source we hold that has real per-cell traffic over
time on a real spatial layout, and it already splits traffic into three
services. The mapping to slices is an explicit modelling assumption:

    internet activity -> eMBB demand
    call activity     -> URLLC demand (voice is the latency-critical service here)
    SMS activity      -> mMTC demand  (small, sporadic messages)

Cells are laid out row-major on a 100 x 100 grid (cellId 1..10000), so a cell's
grid neighbours are its real geographic neighbours.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from milan_features import aggregate_one_day  # noqa: E402

SLICES = ("eMBB", "URLLC", "mMTC")
GRID = 100
CENTER = (50, 50)  # central Milan, the densest part of the grid
DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "data" / "processed" / "unified" / "milan_block.npz"


def block_cells(size: int, center: tuple[int, int] = CENTER) -> list[int]:
    """CellIds of a size x size block centred on `center` (row, col)."""
    r0, c0 = center[0] - size // 2, center[1] - size // 2
    return [(r0 + dr) * GRID + (c0 + dc) + 1 for dr in range(size) for dc in range(size)]


def grid_neighbours(cell_ids: list[int]) -> list[list[int]]:
    """4-connected grid neighbours, as indices into `cell_ids`."""
    index = {cid: i for i, cid in enumerate(cell_ids)}
    out = []
    for cid in cell_ids:
        row, col = divmod(cid - 1, GRID)
        cand = []
        if col > 0:
            cand.append(cid - 1)
        if col < GRID - 1:
            cand.append(cid + 1)
        if row > 0:
            cand.append(cid - GRID)
        if row < GRID - 1:
            cand.append(cid + GRID)
        out.append([index[c] for c in cand if c in index])
    return out


def load_block_activity(data_dir: str, days: int = 7, block_size: int = 10,
                        cache: Path = DEFAULT_CACHE) -> dict:
    """Return real activity [T, N, 3] (internet, call, sms) for the block, cached."""
    cells = block_cells(block_size)
    if cache.exists():
        z = np.load(cache)
        if int(z["days"]) == days and list(z["cells"]) == cells:
            return {k: z[k] for k in z.files}
    files = sorted(Path(data_dir).glob("sms-call-internet-mi-*.txt"))[:days]
    if len(files) < days:
        raise FileNotFoundError(f"Need {days} daily Milan files under {data_dir}, found {len(files)}")
    cell_set = set(cells)
    per_day = []
    for path in files:
        t0 = time.time()
        day = aggregate_one_day(str(path))
        day = day[day["square_id"].isin(cell_set)]
        per_day.append(day)
        print(f"  loaded {path.name}: {len(day):,} block rows ({time.time() - t0:.1f}s)", flush=True)
    import pandas as pd
    frame = pd.concat(per_day, ignore_index=True)
    timestamps = np.sort(frame["time_interval_ms"].unique())
    t_index = {t: i for i, t in enumerate(timestamps)}
    c_index = {c: i for i, c in enumerate(cells)}
    activity = np.zeros((len(timestamps), len(cells), 3), dtype=np.float32)
    ti = frame["time_interval_ms"].map(t_index).to_numpy()
    ci = frame["square_id"].map(c_index).to_numpy()
    activity[ti, ci, 0] = frame["internet_traffic"].to_numpy()
    activity[ti, ci, 1] = (frame["call_in"] + frame["call_out"]).to_numpy()
    activity[ti, ci, 2] = (frame["sms_in"] + frame["sms_out"]).to_numpy()
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, activity=activity, timestamps=timestamps,
                        cells=np.array(cells), days=np.array(days), block_size=np.array(block_size))
    return {"activity": activity, "timestamps": timestamps, "cells": np.array(cells),
            "days": np.array(days), "block_size": np.array(block_size)}


def sub_block(data: dict, size: int) -> tuple[np.ndarray, list[int]]:
    """Activity for a smaller centred block nested inside the cached block."""
    full = [int(c) for c in data["cells"]]
    wanted = block_cells(size)
    idx = [full.index(c) for c in wanted]
    return data["activity"][:, idx, :], wanted


if __name__ == "__main__":
    d = load_block_activity(sys.argv[1] if len(sys.argv) > 1 else
                            r"E:\FYP DATA\6G\data\raw\milan_telecom_italia_full\data")
    a = d["activity"]
    print(f"activity {a.shape} (intervals x cells x [internet, call, sms])")
    print("zero-activity fraction per channel:", np.round((a == 0).mean(axis=(0, 1)), 4))
