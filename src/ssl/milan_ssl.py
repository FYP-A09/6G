"""Milan-specific masked reconstruction and 64-D embedding export."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from .milan_adapter import MilanPreprocessor, aggregate_milan_csv, split_milan_by_time
except ImportError:
    from milan_adapter import MilanPreprocessor, aggregate_milan_csv, split_milan_by_time

MILAN_FEATURE_COUNT = 5
MILAN_EMBEDDING_DIM = 64


@dataclass
class MilanSSLConfig:
    mask_ratio: float = 0.4
    embedding_dim: int = MILAN_EMBEDDING_DIM
    hidden_dim: int = 64
    learning_rate: float = 1e-3
    epochs: int = 5
    batch_size: int = 1024
    validation_fraction: float = 0.2
    seed: int = 0
    device: str = "auto"


@dataclass
class MilanModelOutput:
    embedding: torch.Tensor
    reconstruction: torch.Tensor


class MilanMaskedAutoencoder(nn.Module):
    """Five-feature Milan encoder with a masked reconstruction decoder."""

    def __init__(self, config: MilanSSLConfig | None = None) -> None:
        super().__init__()
        config = config or MilanSSLConfig()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(MILAN_FEATURE_COUNT, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, MILAN_FEATURE_COUNT),
        )

    def forward(self, masked_features: torch.Tensor) -> MilanModelOutput:
        if masked_features.ndim != 2 or masked_features.shape[1] != MILAN_FEATURE_COUNT:
            raise ValueError(f"masked_features must have shape [batch, {MILAN_FEATURE_COUNT}]")
        embedding = self.encoder(masked_features)
        return MilanModelOutput(embedding, self.decoder(embedding))

    @staticmethod
    def reconstruction_loss(
        output: MilanModelOutput,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if output.reconstruction.shape != targets.shape or targets.shape != mask.shape:
            raise ValueError("reconstruction, targets, and mask must have identical shapes")
        errors = (output.reconstruction - targets).pow(2)[mask]
        if errors.numel() == 0:
            raise ValueError("mask must select at least one Milan feature")
        return errors.mean()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(requested)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return result


def _masked_tensors(
    features: torch.Tensor, mask_ratio: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 < mask_ratio <= 1.0:
        raise ValueError("mask_ratio must be greater than 0 and at most 1")
    generator = torch.Generator().manual_seed(seed)
    mask = torch.rand(features.shape, generator=generator) < mask_ratio
    empty = ~mask.any(dim=1)
    if empty.any():
        rows = empty.nonzero(as_tuple=False).flatten()
        columns = torch.randint(features.shape[1], (len(rows),), generator=generator)
        mask[rows, columns] = True
    return features.masked_fill(mask, 0.0), mask


def _loader(
    preprocessor: MilanPreprocessor,
    frame: pd.DataFrame,
    config: MilanSSLConfig,
    seed: int,
    shuffle: bool,
) -> DataLoader:
    batch = preprocessor.transform(frame)
    masked, mask = _masked_tensors(batch.features, config.mask_ratio, seed)
    dataset = TensorDataset(masked, batch.features, mask)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle)


def _run_epoch(
    model: MilanMaskedAutoencoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    model.train(optimizer is not None)
    total_loss = 0.0
    target_count = 0
    for masked, targets, mask in loader:
        masked, targets, mask = masked.to(device), targets.to(device), mask.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        output = model(masked)
        loss = model.reconstruction_loss(output, targets, mask)
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        count = int(mask.sum().item())
        total_loss += loss.item() * count
        target_count += count
    if target_count == 0:
        raise RuntimeError("No masked Milan targets were produced")
    return total_loss / target_count


def _load_frames(data_dir: str | Path, max_files: int | None, max_rows: int | None) -> pd.DataFrame:
    directory = Path(data_dir)
    files = sorted(directory.glob("sms-call-internet-mi-*.csv"))
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No Milan CSV files found under {directory}")
    frames = [aggregate_milan_csv(path) for path in files]
    combined = pd.concat(frames, ignore_index=True).sort_values(
        ["datetime", "CellID"], kind="stable"
    ).reset_index(drop=True)
    return combined.iloc[:max_rows].copy() if max_rows is not None else combined


def train_milan(
    data_dir: str | Path,
    output_dir: str | Path,
    config: MilanSSLConfig | None = None,
    max_files: int | None = None,
    max_rows: int | None = None,
) -> list[dict]:
    config = config or MilanSSLConfig()
    _seed_everything(config.seed)
    device = _device(config.device)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = _load_frames(data_dir, max_files, max_rows)
    train_frame, validation_frame = split_milan_by_time(frame, config.validation_fraction)
    preprocessor = MilanPreprocessor().fit(train_frame)
    model = MilanMaskedAutoencoder(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    best_loss = float("inf")
    history = []
    for epoch in range(1, config.epochs + 1):
        started = time.perf_counter()
        train_loss = _run_epoch(model, _loader(preprocessor, train_frame, config, config.seed + epoch, True), optimizer, device)
        with torch.no_grad():
            validation_loss = _run_epoch(model, _loader(preprocessor, validation_frame, config, config.seed + 10_000 + epoch, False), None, device)
        record = {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss, "seconds": time.perf_counter() - started}
        history.append(record)
        print(f"epoch {epoch}/{config.epochs}: train_loss={train_loss:.6f} validation_loss={validation_loss:.6f} seconds={record['seconds']:.2f}")
        if validation_loss < best_loss:
            best_loss = validation_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_config": asdict(config),
                "preprocessing": {"traffic_columns": list(preprocessor.traffic_columns), "means": preprocessor._means, "stds": preprocessor._stds, "source_dataset": "milan"},
                "best_validation_loss": best_loss,
            }, output / "best_milan_model.pt")
    (output / "metrics.json").write_text(json.dumps({"rows": len(frame), "train_rows": len(train_frame), "validation_rows": len(validation_frame), "best_validation_loss": best_loss, "epochs": history}, indent=2), encoding="utf-8")
    preprocessor.save_metadata(output / "preprocessing.json")
    return history


def export_milan_embeddings(
    data_dir: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    max_files: int | None = None,
    max_rows: int | None = None,
    batch_size: int = 1024,
    device_name: str = "cpu",
) -> None:
    device = _device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = MilanSSLConfig(**checkpoint["model_config"])
    model = MilanMaskedAutoencoder(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    preprocessor = MilanPreprocessor(checkpoint["preprocessing"]["traffic_columns"])
    preprocessor._means = checkpoint["preprocessing"]["means"]
    preprocessor._stds = checkpoint["preprocessing"]["stds"]
    preprocessor._fitted = True
    frame = _load_frames(data_dir, max_files, max_rows)
    batch = preprocessor.transform(frame)
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(frame), batch_size):
            embeddings.append(model(batch.features[start:start + batch_size].to(device)).embedding.cpu().numpy())
    embedding_array = np.concatenate(embeddings)
    result = pd.DataFrame({"node_id": batch.node_ids, "timestamp": batch.timestamps.numpy(), "source_dataset": "milan"})
    for index in range(embedding_array.shape[1]):
        result[f"embedding_{index:02d}"] = embedding_array[:, index]
    result.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--max-files", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    train_milan(args.data_dir, args.output_dir, MilanSSLConfig(epochs=args.epochs), args.max_files, args.max_rows)
