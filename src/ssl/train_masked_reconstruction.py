"""Reproducible Phase 3 training for the 5G-NIDD SSL encoder."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    from .masked_reconstruction import FlowFeatureEncoder, MaskedReconstructionConfig
    from .preprocessing import NIDDPreprocessor, load_nidd_csv, split_by_sequence
except ImportError:
    from masked_reconstruction import FlowFeatureEncoder, MaskedReconstructionConfig
    from preprocessing import NIDDPreprocessor, load_nidd_csv, split_by_sequence


@dataclass
class TrainingConfig:
    epochs: int = 5
    batch_size: int = 256
    validation_fraction: float = 0.2
    mask_ratio: float = 0.3
    learning_rate: float = 1e-3
    hidden_dim: int = 128
    categorical_embedding_dim: int = 8
    embedding_dim: int = 64
    seed: int = 0
    num_workers: int = 0
    device: str = "auto"


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    validation_loss: float
    seconds: float


def set_seed(seed: int) -> None:
    """Seed all local random sources used by preprocessing and training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _make_loader(
    preprocessor: NIDDPreprocessor,
    frame: pd.DataFrame,
    config: TrainingConfig,
    seed: int,
    shuffle: bool,
) -> DataLoader:
    batch = preprocessor.transform_with_mask(
        frame,
        mask_ratio=config.mask_ratio,
        seed=seed,
    )
    dataset = TensorDataset(
        batch.context,
        batch.numeric,
        batch.targets,
        batch.mask,
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=config.num_workers,
    )


def _run_epoch(
    model: FlowFeatureEncoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_targets = 0

    for context, numeric, targets, mask in loader:
        context = context.to(device)
        numeric = numeric.to(device)
        targets = targets.to(device)
        mask = mask.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)
        output = model(context, numeric)
        loss = model.reconstruction_loss(output, targets, mask)
        if training:
            loss.backward()
            optimizer.step()

        target_count = int(mask.sum().item())
        total_loss += loss.item() * target_count
        total_targets += target_count

    if total_targets == 0:
        raise RuntimeError("Training produced no masked reconstruction targets")
    return total_loss / total_targets


def _checkpoint_payload(
    model: FlowFeatureEncoder,
    preprocessor: NIDDPreprocessor,
    model_config: MaskedReconstructionConfig,
    training_config: TrainingConfig,
    metrics: list[EpochMetrics],
    best_validation_loss: float,
) -> dict:
    return {
        "model_state_dict": model.state_dict(),
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "category_sizes": preprocessor.category_sizes,
        "preprocessing": {
            "context_columns": list(preprocessor.context_columns),
            "masked_columns": list(preprocessor.masked_columns),
            "category_to_id": preprocessor._category_to_id,
            "numeric_medians": preprocessor._numeric_medians,
            "numeric_means": preprocessor._numeric_means,
            "numeric_stds": preprocessor._numeric_stds,
        },
        "metrics": [asdict(item) for item in metrics],
        "best_validation_loss": best_validation_loss,
    }


def train_nidd(
    dataset_path: str | Path,
    output_dir: str | Path,
    config: TrainingConfig | None = None,
    n_rows: int | None = None,
) -> list[EpochMetrics]:
    """Train on NIDD and save the best checkpoint plus metrics and metadata."""
    config = config or TrainingConfig()
    if config.epochs < 1 or config.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    set_seed(config.seed)
    device = resolve_device(config.device)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    frame = load_nidd_csv(dataset_path, n_rows=n_rows)
    train_frame, validation_frame = split_by_sequence(
        frame, validation_fraction=config.validation_fraction
    )
    preprocessor = NIDDPreprocessor().fit(train_frame)
    model_config = MaskedReconstructionConfig(
        mask_ratio=config.mask_ratio,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        categorical_embedding_dim=config.categorical_embedding_dim,
        learning_rate=config.learning_rate,
    )
    model = FlowFeatureEncoder(
        config=model_config,
        category_sizes=preprocessor.category_sizes,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    metrics: list[EpochMetrics] = []
    best_validation_loss = float("inf")
    start = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        train_loader = _make_loader(
            preprocessor, train_frame, config, config.seed + epoch, shuffle=True
        )
        validation_loader = _make_loader(
            preprocessor, validation_frame, config, config.seed + 10_000 + epoch, shuffle=False
        )
        train_loss = _run_epoch(model, train_loader, optimizer, device)
        with torch.no_grad():
            validation_loss = _run_epoch(model, validation_loader, None, device)
        epoch_metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            validation_loss=validation_loss,
            seconds=time.perf_counter() - epoch_start,
        )
        metrics.append(epoch_metrics)
        print(
            f"epoch {epoch}/{config.epochs}: train_loss={train_loss:.6f} "
            f"validation_loss={validation_loss:.6f} "
            f"seconds={epoch_metrics.seconds:.2f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(
                _checkpoint_payload(
                    model,
                    preprocessor,
                    model_config,
                    config,
                    metrics,
                    best_validation_loss,
                ),
                output_path / "best_model.pt",
            )

    summary = {
        "dataset_path": str(dataset_path),
        "device": str(device),
        "n_rows_loaded": len(frame),
        "n_train_rows": len(train_frame),
        "n_validation_rows": len(validation_frame),
        "elapsed_seconds": time.perf_counter() - start,
        "best_validation_loss": best_validation_loss,
        "epochs": [asdict(item) for item in metrics],
        "training_config": asdict(config),
    }
    (output_path / "metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    preprocessor.save_metadata(output_path / "preprocessing.json")
    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_path")
    parser.add_argument("output_dir")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-rows", type=int, default=None)
    args = parser.parse_args()
    train_nidd(
        args.dataset_path,
        args.output_dir,
        TrainingConfig(epochs=args.epochs, batch_size=args.batch_size),
        n_rows=args.n_rows,
    )
