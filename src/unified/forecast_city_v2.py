"""City forecaster v2: the same TGNN (GraphSAGE + GRU), trained properly.

forecast_city.py trained on only 5 days, tested on 2, and reused TGNNPredictor,
whose third output goes through a sigmoid, so mMTC could never be predicted above
the cell's 95th percentile. This version:

  more data     train on days 0-40 (1 Nov - 11 Dec 2013), pick the best epoch on
                days 41-44, and test on days 45-51 (16-22 Dec, a full week)
  daily cycle   each step also sees the cell's demand at the same time yesterday
                (known in advance), plus time of day and a weekend flag
  residual head predicts a correction to the last observed value, unbounded,
                so spikes above the usual level can be forecast
  baselines     persistence, same time yesterday, per-cell time-of-day average over
                the training days, and the 5-day city forecaster (v1)

Run: python src/unified/forecast_city_v2.py
Writes data/processed/unified/tgnn_city_forecaster_v2.pt and reports/unified/forecast_city_v2.json.
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
from unified.forecast import WINDOW  # noqa: E402  (puts src/tgnn on the path)
from model import GraphSAGELayer  # noqa: E402
from unified import forecast_city as v1  # noqa: E402
from unified.data import GRID, block_cells, load_block_activity  # noqa: E402
from unified.env import ModelConfig, demand_scale  # noqa: E402
from unified.showcase import MILAN  # noqa: E402
from unified.twin_dataset import BASE, START, full_grid  # noqa: E402

import datetime as dt

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "data" / "processed" / "unified" / "tgnn_city_forecaster_v2.pt"
DAY = 144
TRAIN_DAYS, VAL_DAYS, TEST_DAYS = (0, 41), (41, 45), (45, 52)
PATCH, F_IN = 32, 9
torch.set_num_threads(8)


class CityTGNN(nn.Module):
    """GraphSAGE over the cell graph at every step, a GRU over time, a residual head."""

    def __init__(self, f_in=F_IN, hidden=64):
        super().__init__()
        self.sage = nn.ModuleList([GraphSAGELayer(f_in, hidden), GraphSAGELayer(hidden, hidden)])
        self.gru = nn.GRUCell(hidden, hidden)
        self.head = nn.Sequential(nn.Linear(hidden + 6, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x, edge_index):
        # x: [N, W, F]; features 0-2 = demand / cell scale, 3-5 = same time yesterday for t+1
        n, w, _ = x.shape
        h = torch.zeros(n, self.gru.hidden_size)
        for k in range(w):
            s = x[:, k]
            for layer in self.sage:
                s = layer(s, edge_index)
            h = self.gru(s, h)
        last, yday = x[:, -1, 0:3], x[:, -1, 3:6]
        return torch.relu(last + self.head(torch.cat([h, last, yday], dim=1)))


def edge_index_for(rows: range, cols: range) -> torch.Tensor:
    R, C = len(rows), len(cols)
    src, dst = [], []
    for r in range(R):
        for c in range(C):
            k = r * C + c
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                if 0 <= r + dr < R and 0 <= c + dc < C:
                    src.append((r + dr) * C + c + dc); dst.append(k)
    return torch.tensor([src, dst])


def weekend(t_abs: np.ndarray) -> np.ndarray:
    return np.array([(START + dt.timedelta(int(t) // DAY)).weekday() >= 5 for t in t_abs], np.float32)


def make_window(D, t, cells, cs, t0):
    """Features for predicting interval t+1 from window ending at t. D is indexed by absolute interval - t0."""
    ts = np.arange(t - WINDOW + 1, t + 1)
    cur = D[ts - t0][:, cells] / cs[cells]                      # [W, n, 3]
    yday = D[ts + 1 - DAY - t0][:, cells] / cs[cells]           # same time yesterday, one step ahead
    minute = ((ts + 1) % DAY) / DAY
    tod = np.stack([np.sin(2 * np.pi * minute), np.cos(2 * np.pi * minute)], 1)[:, None, :].repeat(len(cells), 1)
    wk = weekend(ts + 1)[:, None, None].repeat(len(cells), 1)
    x = np.concatenate([cur, yday, tod, wk], axis=2).astype(np.float32)  # [W, n, 9]
    return torch.from_numpy(x).permute(1, 0, 2).contiguous()


def load_range(grid, scale, a, b):
    return (np.asarray(grid[a:b], dtype=np.float32) * scale).astype(np.float32)


def train(grid, scale, cs, epochs=20, steps=1500, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    t0 = 0
    D = load_range(grid, scale, 0, VAL_DAYS[1] * DAY)          # train + validation days (about 0.8 GB)
    ei = edge_index_for(range(PATCH), range(PATCH))
    model = CityTGNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps)
    fit_t = np.arange(DAY + WINDOW, TRAIN_DAYS[1] * DAY - 1)
    val_t = np.arange(VAL_DAYS[0] * DAY, VAL_DAYS[1] * DAY - 1, 7)
    val_patches = [(r, c) for r, c in ((34, 34), (2, 2), (60, 10), (10, 64), (66, 66), (45, 45))]
    idx = lambda r, c: np.array([(r + i) * GRID + c + j for i in range(PATCH) for j in range(PATCH)])

    def loss_at(cells, t):
        x = make_window(D, t, cells, cs, t0)
        y = torch.from_numpy(D[t + 1 - t0][cells] / cs[cells])
        return nn.functional.smooth_l1_loss(model(x, ei), y, beta=0.1)

    best, best_state, hist = float("inf"), None, []
    for ep in range(epochs):
        a = time.time(); model.train()
        for _ in range(steps):
            r, c = rng.integers(0, GRID - PATCH + 1, 2)
            opt.zero_grad(); loss_at(idx(r, c), int(rng.choice(fit_t))).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        model.eval()
        with torch.no_grad():
            v = float(np.mean([loss_at(idx(r, c), int(t)).item() for r, c in val_patches for t in val_t[::3]]))
        hist.append(round(v, 5))
        if v < best:
            best, best_state = v, {k: p.clone() for k, p in model.state_dict().items()}
        print(f"v2 epoch {ep + 1}/{epochs}: val loss {v:.5f} ({time.time() - a:.0f}s)", flush=True)
    model.load_state_dict(best_state)
    del D
    return model, hist


def predict(model, D, t0, ts, cs, rows=20):
    """forecast for t+1 for every t in ts, all 10,000 cells, in horizontal strips with a one-row halo."""
    out = np.zeros((len(ts), GRID * GRID, 3), np.float32)
    model.eval()
    with torch.no_grad():
        for r0 in range(0, GRID, rows):
            lo, hi = max(0, r0 - 1), min(GRID, r0 + rows + 1)
            cells = np.array([r * GRID + c for r in range(lo, hi) for c in range(GRID)])
            ei = edge_index_for(range(lo, hi), range(GRID))
            keep = slice((r0 - lo) * GRID, (r0 - lo + rows) * GRID)
            for k, t in enumerate(ts):
                out[k, cells[keep]] = (model(make_window(D, t, cells, cs, t0), ei).numpy() * cs[cells])[keep]
    return out


def regions():
    block = np.zeros(GRID * GRID, bool); block[[c - 1 for c in block_cells(10)]] = True
    core = np.zeros(GRID * GRID, bool); core[[c - 1 for c in block_cells(20)]] = True
    return {"block_100": block, "core_ring_300": core & ~block, "outer_9600": ~core, "city_10000": np.ones_like(block)}


def main():
    t_start = time.time()
    cfg = ModelConfig()
    block = load_block_activity(MILAN)
    scale = demand_scale(block["activity"][:720], cfg).astype(np.float32)
    grid = full_grid(BASE / "milan_full_grid_62d.npy")
    # per-cell scale and time-of-day average from the training days only, streamed day by day
    acc, p95_parts = np.zeros((DAY, GRID * GRID, 3)), []
    for d in range(*TRAIN_DAYS):
        day = load_range(grid, scale, d * DAY, (d + 1) * DAY)
        acc += day
        p95_parts.append(np.percentile(day, 95, axis=0))
    tod_mean = (acc / (TRAIN_DAYS[1] - TRAIN_DAYS[0])).astype(np.float32)
    p95 = np.median(np.stack(p95_parts), axis=0)                 # typical daily 95th percentile per cell
    cs = np.maximum(p95, 0.01 * np.percentile(p95, 95, axis=0)).astype(np.float32)
    print(f"stats from {TRAIN_DAYS[1]} training days ({time.time() - t_start:.0f}s)", flush=True)

    model, hist = train(grid, scale, cs)
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "cell_scale": cs}, MODEL)

    t0 = TEST_DAYS[0] * DAY - DAY - WINDOW                       # history for yesterday features
    D = load_range(grid, scale, t0, TEST_DAYS[1] * DAY)
    ts = np.arange(TEST_DAYS[0] * DAY - 1, TEST_DAYS[1] * DAY - 1)  # predict every test interval
    truth = D[ts + 1 - t0]
    a = time.time()
    fc = {"tgnn_v2": predict(model, D, t0, ts, cs)}
    print(f"v2 predictions in {time.time() - a:.0f}s", flush=True)
    fc["persistence"] = D[ts - t0]
    fc["same_time_yesterday"] = D[ts + 1 - DAY - t0]
    fc["time_of_day_average"] = tod_mean[(ts + 1) % DAY]
    m1, cs1 = v1.load()
    f1 = v1.predict(m1, D, cs1, t0=t0)                           # v1 forecast[t] = prediction of t+1
    fc["tgnn_v1_5days"] = f1[ts - t0].astype(np.float32)

    report = {"split_days": {"train": TRAIN_DAYS, "validation": VAL_DAYS, "test": TEST_DAYS,
                             "test_dates": f"{START + dt.timedelta(TEST_DAYS[0])} to {START + dt.timedelta(TEST_DAYS[1] - 1)}"},
              "val_history": hist, "regions": {}}
    for name, m in regions().items():
        rep = {}
        for k, f in fc.items():
            err = f[:, m] - truth[:, m]
            rep[k] = {"mse": (err ** 2).mean(axis=(0, 1)).round(4).tolist(), "mae": np.abs(err).mean(axis=(0, 1)).round(4).tolist()}
        for k in fc:
            rep[k]["better_than_persistence"] = (1 - np.array(rep[k]["mse"]) / np.array(rep["persistence"]["mse"])).round(4).tolist()
            rep[k]["better_than_best_simple"] = (1 - np.array(rep[k]["mse"]) / np.minimum.reduce(
                [np.array(rep[b]["mse"]) for b in ("persistence", "same_time_yesterday", "time_of_day_average")])).round(4).tolist()
        report["regions"][name] = rep
    report["seconds"] = round(time.time() - t_start)
    out = ROOT / "reports" / "unified" / "forecast_city_v2.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, rep in report["regions"].items():
        print(name, {k: v["better_than_persistence"] for k, v in rep.items()})
    print("wrote", out)


if __name__ == "__main__":
    main()
