"""Real next-step traffic-forecasting training for TGNNPredictor on Milan data.

Addresses Remaining Work item 6 ("retrain the TGNN on real trained SSL embeddings
instead of placeholders"). Every embedding, adjacency edge, and target value here
comes from the real Milan release — nothing is synthetic.

Pipeline:
  1. Load real Milan rows for a compact, contiguous sub-grid of cells (dense
     [N, N] adjacency only stays tractable at this scale — TGNNPredictor's
     current API is dense-adjacency, see model.py).
  2. Run each row through the already-trained Milan SSL encoder
     (milan_ssl.MilanMaskedAutoencoder) to get real 64-D embeddings — the same
     checkpoint tgnn_bridge.py consumes for the single-window smoke test.
  3. Build the real grid adjacency: Milan's 10,000 cells are laid out row-major
     in a 100x100 grid (confirmed against milano-grid.geojson — cellId N's
     immediate grid neighbors are N-1, N+1, N-100, N+100). This is the one
     graph structure the release actually supports; B5G/NeversNet5G-style
     routing topology doesn't apply here.
  4. Slide a window of T consecutive real timesteps across the real time series
     and predict the (T+1)-th real, min-max-normalized internet-traffic value
     per cell — trained against TGNNPredictor's existing sigmoid-bounded
     "predicted_load" output head (the natural fit for a real [0,1] target).
  5. Train with real backprop (MSE loss), chronological train/validation split.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ssl"))

from model import TGNNPredictor  # noqa: E402
from milan_ssl import MilanMaskedAutoencoder, MilanSSLConfig, _load_frames  # noqa: E402
from milan_adapter import MilanPreprocessor  # noqa: E402

NAVY = "#1F3864"
RED = "#B5433A"
GRAY = "#595959"


@dataclass
class ForecastConfig:
    grid_size: int = 20          # 20x20 sub-grid = 400 real cells
    grid_origin: int = 4040      # top-left cellId of the sub-grid (1-indexed, row-major)
    window_length: int = 5       # T real consecutive timesteps as input
    epochs: int = 15
    learning_rate: float = 5e-4
    validation_fraction: float = 0.2
    seed: int = 0


def _sub_grid_cell_ids(origin: int, size: int) -> list[int]:
    """Real cellIds for a size x size contiguous block starting at `origin`,
    using the confirmed row-major 100x100 layout (cellId 1..10000)."""
    origin_row, origin_col = divmod(origin - 1, 100)
    ids = []
    for dr in range(size):
        for dc in range(size):
            row, col = origin_row + dr, origin_col + dc
            if row >= 100 or col >= 100:
                raise ValueError("sub-grid exceeds the real 100x100 Milan grid bounds")
            ids.append(row * 100 + col + 1)
    return ids


def _grid_adjacency(cell_ids: list[int]) -> torch.Tensor:
    """Real 4-connectivity grid adjacency: cellId N's neighbors are N-1, N+1,
    N-100, N+100, restricted to row/column boundaries so it never wraps."""
    index = {cid: i for i, cid in enumerate(cell_ids)}
    n = len(cell_ids)
    adj = torch.zeros(n, n)
    for cid, i in index.items():
        row, col = divmod(cid - 1, 100)
        candidates = []
        if col > 0:
            candidates.append(cid - 1)
        if col < 99:
            candidates.append(cid + 1)
        candidates.append(cid - 100)
        candidates.append(cid + 100)
        for neighbor in candidates:
            j = index.get(neighbor)
            if j is not None:
                adj[i, j] = 1.0
                adj[j, i] = 1.0
    return adj


def _load_milan_checkpoint(path: str) -> tuple[MilanMaskedAutoencoder, MilanPreprocessor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = MilanSSLConfig(**checkpoint["model_config"])
    model = MilanMaskedAutoencoder(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    preprocessor = MilanPreprocessor(checkpoint["preprocessing"]["traffic_columns"])
    preprocessor._means = checkpoint["preprocessing"]["means"]
    preprocessor._stds = checkpoint["preprocessing"]["stds"]
    preprocessor._fitted = True
    return model, preprocessor


def build_dataset(
    milan_data_dir: str,
    ssl_checkpoint_path: str,
    config: ForecastConfig,
    max_files: int,
    max_rows: int | None,
):
    cell_ids = _sub_grid_cell_ids(config.grid_origin, config.grid_size)
    cell_id_strs = [str(c) for c in cell_ids]
    frame = _load_frames(milan_data_dir, max_files=max_files, max_rows=max_rows)
    frame = frame[frame["CellID"].isin(cell_id_strs)].copy()
    if frame.empty:
        raise RuntimeError("No real rows found for the requested sub-grid — check grid_origin")

    ssl_model, ssl_preprocessor = _load_milan_checkpoint(ssl_checkpoint_path)
    batch = ssl_preprocessor.transform(frame)
    with torch.no_grad():
        embeddings = ssl_model(batch.features).embedding  # [rows, 64], real trained SSL output

    frame = frame.reset_index(drop=True)
    frame["_row"] = range(len(frame))
    timestamps = sorted(frame["datetime"].unique().tolist())
    n_cells, n_time = len(cell_ids), len(timestamps)
    if n_time < config.window_length + 2:
        raise RuntimeError(
            f"Only {n_time} real timestamps available in this slice — need at least "
            f"{config.window_length + 2} for a train/validation window split. Increase --max-rows."
        )

    cell_index = {c: i for i, c in enumerate(cell_id_strs)}
    time_index = {t: i for i, t in enumerate(timestamps)}
    embed_grid = torch.zeros(n_cells, n_time, embeddings.shape[1])
    internet_raw = torch.zeros(n_cells, n_time)
    present = torch.zeros(n_cells, n_time, dtype=torch.bool)
    for _, row in frame.iterrows():
        ci, ti = cell_index[row["CellID"]], time_index[row["datetime"]]
        embed_grid[ci, ti] = embeddings[row["_row"]]
        internet_raw[ci, ti] = float(row["internet"])
        present[ci, ti] = True
    if not present.all():
        missing = int((~present).sum())
        raise RuntimeError(
            f"{missing} (cell, timestamp) pairs are missing in this real slice — "
            "the sub-grid/time range isn't fully rectangular; pick a smaller grid_size "
            "or a range with complete coverage."
        )

    adjacency = _grid_adjacency(cell_ids)
    return embed_grid, internet_raw, adjacency, cell_ids, timestamps


def make_windows(
    embed_grid: torch.Tensor, internet_raw: torch.Tensor, window_length: int
):
    """[N, T, D] embeddings + [N, T] real internet traffic -> list of
    (embeddings_window [N, window_length, D], target [N]) pairs, target = the
    real min-max-normalized next-step traffic value."""
    n, t, d = embed_grid.shape
    windows = []
    for start in range(0, t - window_length):
        target_index = start + window_length
        windows.append((embed_grid[:, start:target_index, :], internet_raw[:, target_index]))
    return windows


def train(config: ForecastConfig, milan_data_dir: str, ssl_checkpoint: str,
          output_dir: str, max_files: int, max_rows: int | None):
    torch.manual_seed(config.seed)
    embed_grid, internet_raw, adjacency, cell_ids, timestamps = build_dataset(
        milan_data_dir, ssl_checkpoint, config, max_files, max_rows
    )
    print(f"Real dataset: {len(cell_ids)} cells x {len(timestamps)} real timesteps, "
          f"{int(adjacency.sum().item())} real grid edges")

    windows = make_windows(embed_grid, internet_raw, config.window_length)
    split = max(1, int(len(windows) * (1 - config.validation_fraction)))
    train_windows, validation_windows = windows[:split], windows[split:]
    print(f"{len(train_windows)} train windows, {len(validation_windows)} validation windows "
          f"(chronological split, real held-out future)")

    # Real min-max target normalization fit on TRAIN windows only.
    train_targets = torch.cat([target for _, target in train_windows])
    target_min, target_max = float(train_targets.min()), float(train_targets.max())
    target_range = max(target_max - target_min, 1e-6)

    def normalize(target: torch.Tensor) -> torch.Tensor:
        return ((target - target_min) / target_range).clamp(0.0, 1.0)

    model = TGNNPredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    def run_epoch(window_set, training: bool) -> float:
        model.train(training)
        total_loss, total_count = 0.0, 0
        for embeddings_window, raw_target in window_set:
            target = normalize(raw_target)
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(embeddings_window, adjacency)
            predicted_load = prediction[:, 2]
            loss = nn.functional.mse_loss(predicted_load, target)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(target)
            total_count += len(target)
        return total_loss / max(total_count, 1)

    history = []
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    best_validation = float("inf")
    start = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        train_mse = run_epoch(train_windows, training=True)
        with torch.no_grad():
            validation_mse = run_epoch(validation_windows, training=False)
        record = {
            "epoch": epoch, "train_mse": train_mse, "validation_mse": validation_mse,
            "seconds": time.perf_counter() - epoch_start,
        }
        history.append(record)
        print(f"epoch {epoch}/{config.epochs}: train_mse={train_mse:.6f} "
              f"validation_mse={validation_mse:.6f} seconds={record['seconds']:.2f}")
        if validation_mse < best_validation:
            best_validation = validation_mse
            torch.save({"model_state_dict": model.state_dict(),
                        "target_min": target_min, "target_max": target_max,
                        "config": asdict(config)}, output_path / "best_tgnn_milan.pt")

    elapsed = time.perf_counter() - start
    summary = {
        "n_cells": len(cell_ids), "n_timesteps": len(timestamps),
        "n_train_windows": len(train_windows), "n_validation_windows": len(validation_windows),
        "best_validation_mse": best_validation, "elapsed_seconds": elapsed,
        "config": asdict(config), "epochs": history,
    }
    (output_path / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 5))
    epochs_x = [r["epoch"] for r in history]
    ax.plot(epochs_x, [r["train_mse"] for r in history], color=NAVY, marker="o", label="train MSE")
    ax.plot(epochs_x, [r["validation_mse"] for r in history], color=RED, marker="o", label="validation MSE (real held-out future)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE (normalized load, [0,1] scale)")
    ax.set_title(f"Real TGNN Training — Next-Step Traffic Forecast on {len(cell_ids)} Real Milan Cells",
                 fontsize=12.5, fontweight="bold", color=NAVY)
    ax.legend(); ax.grid(alpha=0.25); ax.set_facecolor("#FAFAFA")
    fig.tight_layout()
    figures_dir = Path(__file__).resolve().parent.parent.parent / "docs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "tgnn_milan_forecast_training.png", dpi=200)
    plt.close(fig)
    print(f"\nSaved checkpoint, metrics, and training-curve figure under {output_path} "
          f"and docs/figures/tgnn_milan_forecast_training.png")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("milan_data_dir")
    parser.add_argument("ssl_checkpoint")
    parser.add_argument("output_dir")
    parser.add_argument("--max-files", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--grid-origin", type=int, default=4040)
    parser.add_argument("--window-length", type=int, default=5)
    args = parser.parse_args()
    train(
        ForecastConfig(
            grid_size=args.grid_size, grid_origin=args.grid_origin,
            window_length=args.window_length, epochs=args.epochs,
        ),
        args.milan_data_dir, args.ssl_checkpoint, args.output_dir,
        args.max_files, args.max_rows,
    )
