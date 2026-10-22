"""Does self-supervised pretraining help the city forecaster?  (the Milan side of the SSL question)

The 5G-NIDD study (ssl_nidd.py) found no gain from pretraining. The earlier Milan result (SSL embeddings = raw features) came from a one-day,
one-seed TGNN. Here the question is asked of the real city forecaster (forecast_city_v2.py), with the same data, split and budget:

  raw   the v2 forecaster as it is
  ssl   the same forecaster plus a frozen 16-d embedding of each cell's input window, produced by an encoder that was pretrained
        WITHOUT labels by masked reconstruction (30% of the entries of a window are hidden and rebuilt from the rest) on windows from the
        training days only; the embedding goes to the output head next to the GRU state

Both are trained for the same number of steps on days 0-40 (days 5 and 6 excluded) and tested on the held-out week, every cell.

Run: python src/unified/ssl_milan_forecast.py <raw|ssl> <seed> [epochs]   -> reports/unified/ssl_milan_<variant>_<seed>.json
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
from unified import forecast_city_v2 as v2  # noqa: E402
from unified.data import GRID, load_block_activity  # noqa: E402
from unified.env import ModelConfig, demand_scale  # noqa: E402
from unified.showcase import MILAN  # noqa: E402
from unified.twin_dataset import BASE, full_grid  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DAY, WINDOW, PATCH = v2.DAY, v2.WINDOW, v2.PATCH
EMB = 16
torch.set_num_threads(3)


class WindowEncoder(nn.Module):
    def __init__(self, d_in=WINDOW * v2.F_IN):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, 128), nn.ReLU(), nn.Linear(128, EMB))
        self.dec = nn.Sequential(nn.Linear(EMB, 128), nn.ReLU(), nn.Linear(128, d_in))

    def forward(self, x):
        return self.enc(x)


class CityTGNNSSL(v2.CityTGNN):
    def __init__(self, encoder, hidden=64):
        super().__init__(hidden=hidden)
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.head = nn.Sequential(nn.Linear(hidden + 6 + EMB, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x, edge_index):
        n, w, _ = x.shape
        h = torch.zeros(n, self.gru.hidden_size)
        for k in range(w):
            s = x[:, k]
            for layer in self.sage:
                s = layer(s, edge_index)
            h = self.gru(s, h)
        last, yday = x[:, -1, 0:3], x[:, -1, 3:6]
        with torch.no_grad():
            emb = self.encoder(x.reshape(n, -1))
        return torch.relu(last + self.head(torch.cat([h, last, yday, emb], dim=1)))


def pretrain(D, cs, seed, steps=3000):
    """Masked reconstruction of input windows (no targets, training days only)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    enc = WindowEncoder()
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
    fit_t = np.array([t for t in range(DAY + WINDOW, v2.TRAIN_DAYS[1] * DAY - 1) if (t + 1) // DAY not in v2.EXCLUDE_DAYS])
    hist = []
    for i in range(steps):
        r, c = rng.integers(0, GRID - PATCH + 1, 2)
        cells = np.array([(r + a) * GRID + c + b for a in range(PATCH) for b in range(PATCH)])
        x = v2.make_window(D, int(rng.choice(fit_t)), cells, cs, 0).reshape(len(cells), -1)
        mask = (torch.rand_like(x) < 0.3).float()
        loss = (((enc.dec(enc.enc(x * (1 - mask))) - x) ** 2) * mask).sum() / mask.sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); opt.step()
        if (i + 1) % 500 == 0:
            hist.append(round(float(loss), 5))
            print(f"  ssl pretrain step {i + 1}: masked reconstruction loss {hist[-1]}", flush=True)
    return enc, hist


def main(variant, seed, epochs):
    t_start = time.time()
    cfg = ModelConfig()
    block = load_block_activity(MILAN)
    scale = demand_scale(block["activity"][:720], cfg).astype(np.float32)
    grid = full_grid(BASE / "milan_full_grid_62d.npy")
    # same per-cell scale as forecast_city_v2
    ck = torch.load(v2.MODEL, weights_only=False)
    cs = ck["cell_scale"]
    D = v2.load_range(grid, scale, 0, v2.VAL_DAYS[1] * DAY) if variant == "ssl" else None
    hist = None
    torch.manual_seed(seed)
    if variant == "ssl":
        enc, hist = pretrain(D, cs, seed)
        del D
        model = CityTGNNSSL(enc)
    else:
        model = v2.CityTGNN()
    # reuse the v2 training loop by temporarily swapping the model constructor
    orig = v2.CityTGNN
    v2.CityTGNN = lambda *a, **k: model
    try:
        model, val_hist = v2.train(grid, scale, cs, epochs=epochs, steps=1500, seed=seed)
    finally:
        v2.CityTGNN = orig
    t0 = v2.TEST_DAYS[0] * DAY - DAY - WINDOW
    Dt = v2.load_range(grid, scale, t0, v2.TEST_DAYS[1] * DAY)
    ts = np.arange(v2.TEST_DAYS[0] * DAY - 1, v2.TEST_DAYS[1] * DAY - 1)
    truth = Dt[ts + 1 - t0]
    fc = v2.predict(model, Dt, t0, ts, cs)
    pers = Dt[ts - t0]
    mse = ((fc - truth) ** 2).mean(axis=(0, 1)); pm = ((pers - truth) ** 2).mean(axis=(0, 1))
    out = {"variant": variant, "seed": seed, "epochs": epochs, "val_history": val_hist, "pretrain_loss": hist, "mse": mse.round(4).tolist(), "persistence_mse": pm.round(4).tolist(),
           "better_than_persistence": (1 - mse / pm).round(4).tolist(), "seconds": round(time.time() - t_start)}
    (ROOT / "reports" / "unified" / f"ssl_milan_{variant}_{seed}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(out["better_than_persistence"], flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 8)
