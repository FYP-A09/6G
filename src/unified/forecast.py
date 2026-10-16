"""Next-interval per-slice demand forecast for the unified environment, using the
project's TGNNPredictor (GraphSAGE + GRU) on the real grid adjacency.

Trained on the training days only; produces forecast[t] = prediction of demand[t+1]
made with data up to t, which the environment and the agents consume.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tgnn"))
from model import TGNNPredictor  # noqa: E402

WINDOW = 6  # one hour of 10-minute history


def _features(demand: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """[T, N, 5]: per-slice demand / training max, plus time-of-day sin/cos."""
    T, N, _ = demand.shape
    minute = (np.arange(T) % 144) / 144.0
    tod = np.stack([np.sin(2 * np.pi * minute), np.cos(2 * np.pi * minute)], axis=1)
    return np.concatenate([demand / scale, np.broadcast_to(tod[:, None, :], (T, N, 2))], axis=2).astype(np.float32)


def train_forecaster(demand: np.ndarray, neighbours: list[list[int]], train_end: int,
                     epochs: int = 30, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    T, N, _ = demand.shape
    scale = np.maximum(demand[:train_end].max(axis=(0, 1)), 1e-6)
    x = torch.from_numpy(_features(demand, scale))
    y = torch.from_numpy((demand / scale).astype(np.float32))
    adj = torch.zeros(N, N)
    for i, nb in enumerate(neighbours):
        adj[i, nb] = 1.0
    model = TGNNPredictor(embedding_dim=5)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_t = list(range(WINDOW - 1, train_end - 1))
    val_cut = int(len(train_t) * 0.85)
    fit_t, val_t = train_t[:val_cut], train_t[val_cut:]

    def window(t):
        return x[t - WINDOW + 1:t + 1].permute(1, 0, 2)  # [N, W, 5]

    def loss_on(ts):
        with torch.no_grad():
            return float(np.mean([nn.functional.mse_loss(model(window(t), adj), y[t + 1]).item() for t in ts]))

    best, best_state, history = float("inf"), None, []
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        t0 = time.time()
        for t in rng.permutation(fit_t):
            opt.zero_grad()
            nn.functional.mse_loss(model(window(t), adj), y[t + 1]).backward()
            opt.step()
        v = loss_on(val_t)
        history.append(v)
        if v < best:
            best, best_state = v, {k: p.clone() for k, p in model.state_dict().items()}
        print(f"forecaster epoch {epoch + 1}/{epochs}: val mse {v:.5f} ({time.time() - t0:.1f}s)", flush=True)
    model.load_state_dict(best_state)
    return {"forecast": predict(model, demand, neighbours, scale), "val_history": history,
            "scale": scale, "model": model}


def predict(model: TGNNPredictor, demand: np.ndarray, neighbours: list[list[int]], scale: np.ndarray) -> np.ndarray:
    """forecast[t] = prediction of demand[t+1] from the window ending at t (persistence before the first full window)."""
    T, N, _ = demand.shape
    x = torch.from_numpy(_features(demand, scale))
    adj = torch.zeros(N, N)
    for i, nb in enumerate(neighbours):
        adj[i, nb] = 1.0
    forecast = demand.copy().astype(np.float64)
    model.eval()
    with torch.no_grad():
        for t in range(WINDOW - 1, T - 1):
            forecast[t] = model(x[t - WINDOW + 1:t + 1].permute(1, 0, 2), adj).numpy() * scale
    return forecast


def forecast_errors(demand: np.ndarray, forecast: np.ndarray, start: int, end: int) -> dict:
    """MSE (per slice, in Mbps^2) of the TGNN forecast and of persistence over [start, end)."""
    ts = np.arange(start, end - 1)
    truth = demand[ts + 1]
    tg = ((forecast[ts] - truth) ** 2).mean(axis=(0, 1))
    pe = ((demand[ts] - truth) ** 2).mean(axis=(0, 1))
    return {"tgnn_mse": tg.tolist(), "persistence_mse": pe.tolist(),
            "tgnn_better_by": (1 - tg / np.maximum(pe, 1e-12)).tolist()}
