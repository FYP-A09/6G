"""A city-wide next-10-minute forecaster: the same TGNNPredictor, trained so it transfers.

The block forecaster (forecast.py) is trained on the 100 central cells with one global
scale per slice. Applied to all 10,000 cells it over-predicts the quiet outer cells by
about 6 Mbps (it learned the centre's traffic level) and is 7x worse than persistence
there. Two changes fix the cause:

  per-cell scaling   each cell's traffic is divided by its own training-period 95th
                     percentile, so the model learns the shape of change, not the level
  spatial sampling   every step trains on a random 32 x 32 patch of the real grid, so
                     quiet suburbs, the centre and everything between are all seen

Same split as the experiments: train on the first 5 real days (intervals < 720),
evaluate on the next 2 (720-1007), on all 10,000 cells, against persistence.

Run: python src/unified/forecast_city.py
Writes data/processed/unified/tgnn_city_forecaster.pt and reports/unified/forecast_city.json.
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
from unified.forecast import WINDOW, predict as predict_block  # noqa: E402  (puts src/tgnn on the path)
from model import TGNNPredictor  # noqa: E402
from unified.data import GRID, block_cells, grid_neighbours, load_block_activity  # noqa: E402
from unified.env import ModelConfig, demand_scale  # noqa: E402
from unified.showcase import MILAN, MODEL as BLOCK_MODEL  # noqa: E402
from unified.twin_dataset import BASE, full_grid  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "data" / "processed" / "unified" / "tgnn_city_forecaster.pt"
TRAIN_END, TEST_END, PATCH = 720, 1008, 32


def city_demand(end: int = TEST_END) -> np.ndarray:
    """[end, 10000, 3] Mbps on the same scale as every other unified product."""
    block = load_block_activity(MILAN)
    scale = demand_scale(block["activity"][:TRAIN_END], ModelConfig())
    grid = full_grid(BASE / "milan_full_grid_62d.npy")
    return (np.array(grid[:end], dtype=np.float32) * scale).astype(np.float32)


def cell_scales(demand: np.ndarray) -> np.ndarray:
    """Per-cell, per-slice training 95th percentile, floored at 1% of the slice's city-wide value."""
    p95 = np.percentile(demand[:TRAIN_END], 95, axis=0)
    return np.maximum(p95, 0.01 * np.percentile(demand[:TRAIN_END], 95, axis=(0, 1))).astype(np.float32)


def features(demand: np.ndarray, cs: np.ndarray, t0: int = 0) -> np.ndarray:
    T, N, _ = demand.shape
    minute = ((np.arange(T) + t0) % 144) / 144.0
    tod = np.stack([np.sin(2 * np.pi * minute), np.cos(2 * np.pi * minute)], axis=1).astype(np.float32)
    return np.concatenate([demand / cs, np.broadcast_to(tod[:, None, :], (T, N, 2))], axis=2).astype(np.float32)


def patch_idx(r0: int, c0: int, size: int) -> list[int]:
    return [(r0 + r) * GRID + c0 + c for r in range(size) for c in range(size)]


def patch_adj(r0: int, c0: int, size: int) -> tuple[list[int], torch.Tensor]:
    idx = patch_idx(r0, c0, size)
    adj = torch.zeros(size * size, size * size)
    for k in range(size * size):
        r, c = divmod(k, size)
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            if 0 <= r + dr < size and 0 <= c + dc < size:
                adj[k, (r + dr) * size + c + dc] = 1.0
    return idx, adj


def train(demand: np.ndarray, cs: np.ndarray, epochs: int = 12, steps: int = 1500, seed: int = 0) -> tuple[TGNNPredictor, list]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x = torch.from_numpy(features(demand[:TRAIN_END], cs))
    y = torch.from_numpy(demand[:TRAIN_END] / cs)
    model = TGNNPredictor(embedding_dim=5)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ts = np.arange(WINDOW - 1, TRAIN_END - 1)
    fit_t, val_t = ts[:int(len(ts) * 0.85)], ts[int(len(ts) * 0.85):]
    _, adj = patch_adj(0, 0, PATCH)
    val_patches = [patch_idx(r, c, PATCH) for r, c in ((34, 34), (2, 2), (60, 10), (10, 64), (66, 66))]

    def loss_at(idx, t):
        return nn.functional.mse_loss(model(x[t - WINDOW + 1:t + 1, idx].permute(1, 0, 2), adj), y[t + 1, idx])

    best, best_state, history = float("inf"), None, []
    for ep in range(epochs):
        t0 = time.time()
        model.train()
        for _ in range(steps):
            r0, c0 = rng.integers(0, GRID - PATCH + 1, 2)
            idx = patch_idx(int(r0), int(c0), PATCH)
            opt.zero_grad()
            loss_at(idx, int(rng.choice(fit_t))).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(np.mean([loss_at(idx, int(t)).item() for idx in val_patches for t in val_t[::4]]))
        history.append(v)
        if v < best:
            best, best_state = v, {k: p.clone() for k, p in model.state_dict().items()}
        print(f"city forecaster epoch {ep + 1}/{epochs}: val mse {v:.4f} ({time.time() - t0:.0f}s)", flush=True)
    model.load_state_dict(best_state)
    return model, history


def predict(model: TGNNPredictor, demand: np.ndarray, cs: np.ndarray, t0: int = 0, rows: int = 25) -> np.ndarray:
    """forecast[t] = prediction of demand[t+1] for every cell, run in horizontal strips of the grid
    (with a one-row halo, so every cell sees all four neighbours) to keep memory small."""
    T, N, _ = demand.shape
    x = torch.from_numpy(features(demand, cs, t0))
    out = demand.astype(np.float64).copy()
    model.eval()
    with torch.no_grad():
        for r0 in range(0, GRID, rows):
            lo, hi = max(0, r0 - 1), min(GRID, r0 + rows + 1)
            idx = [r * GRID + c for r in range(lo, hi) for c in range(GRID)]
            keep = slice((r0 - lo) * GRID, (r0 - lo + rows) * GRID)
            nb = grid_neighbours([i + 1 for i in idx])
            adj = torch.zeros(len(idx), len(idx))
            for k, js in enumerate(nb):
                adj[k, js] = 1.0
            sl = idx[keep]
            for t in range(WINDOW - 1, T - 1):
                pred = model(x[t - WINDOW + 1:t + 1, idx].permute(1, 0, 2), adj).numpy()[keep]
                out[t, sl] = pred * cs[sl]
    return out


def regional_errors(fc: np.ndarray, demand: np.ndarray, start: int, end: int) -> dict:
    ts = np.arange(start, end - 1)
    truth, f, p = demand[ts + 1].astype(np.float64), fc[ts], demand[ts].astype(np.float64)
    block = np.zeros(GRID * GRID, bool); block[[c - 1 for c in block_cells(10)]] = True
    core = np.zeros(GRID * GRID, bool); core[[c - 1 for c in block_cells(20)]] = True
    out = {}
    for name, m in (("block_100", block), ("core_ring_300", core & ~block), ("outer_9600", ~core), ("city_10000", np.ones_like(block))):
        e = ((f[:, m] - truth[:, m]) ** 2).mean(axis=(0, 1)); q = ((p[:, m] - truth[:, m]) ** 2).mean(axis=(0, 1))
        out[name] = {"tgnn_mse": e.round(4).tolist(), "persistence_mse": q.round(4).tolist(),
                     "better_than_persistence": (1 - e / q).round(4).tolist(),
                     "bias_mbps": (f[:, m] - truth[:, m]).mean(axis=(0, 1)).round(4).tolist()}
    return out


def load():
    ck = torch.load(MODEL, weights_only=False)
    model = TGNNPredictor(embedding_dim=5)
    model.load_state_dict(ck["state"])
    return model, ck["cell_scale"]


def main() -> None:
    demand = city_demand()
    cs = cell_scales(demand)
    model, history = train(demand, cs)
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "cell_scale": cs}, MODEL)
    t0 = time.time()
    fc = predict(model, demand, cs)
    print(f"predicted 10,000 cells x {len(demand)} intervals in {time.time() - t0:.0f}s", flush=True)
    report = {"split": {"train": [0, TRAIN_END], "test": [TRAIN_END, TEST_END]}, "val_history": history,
              "city_forecaster": regional_errors(fc, demand, TRAIN_END, TEST_END)}
    ck = torch.load(BLOCK_MODEL, weights_only=False)
    bm = TGNNPredictor(embedding_dim=5); bm.load_state_dict(ck["state"])
    lo = TRAIN_END - 144  # whole days, so the block forecaster's time-of-day features line up
    fb = np.zeros_like(fc)
    fb[lo:] = predict_block(bm, demand[lo:].astype(np.float64), grid_neighbours(list(range(1, GRID * GRID + 1))), ck["scale"])
    report["block_forecaster_on_city"] = regional_errors(fb, demand, TRAIN_END, TEST_END)
    out = ROOT / "reports" / "unified" / "forecast_city.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for k in ("city_forecaster", "block_forecaster_on_city"):
        print(k)
        for region, v in report[k].items():
            print(f"  {region}: better than persistence {v['better_than_persistence']}, bias {v['bias_mbps']}")


if __name__ == "__main__":
    main()
