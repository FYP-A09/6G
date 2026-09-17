"""Measure real masked-reconstruction preprocessing and inference overhead."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    from .masked_reconstruction import FlowFeatureEncoder, MaskedReconstructionConfig
    from .preprocessing import NIDDPreprocessor, load_nidd_csv
    from .train_masked_reconstruction import resolve_device
except ImportError:
    from masked_reconstruction import FlowFeatureEncoder, MaskedReconstructionConfig
    from preprocessing import NIDDPreprocessor, load_nidd_csv
    from train_masked_reconstruction import resolve_device


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_trained_model(checkpoint_path: str | Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = MaskedReconstructionConfig(**checkpoint["model_config"])
    model = FlowFeatureEncoder(
        config=model_config,
        category_sizes=checkpoint["category_sizes"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    preprocessor = NIDDPreprocessor.from_metadata(checkpoint["preprocessing"])
    return model, preprocessor, checkpoint


def profile_model(
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    n_rows: int = 2000,
    batch_size: int = 256,
    warmup_batches: int = 1,
    measured_batches: int = 10,
    mask_seed: int = 0,
    device_name: str = "auto",
) -> dict:
    """Profile I/O, preprocessing, and real model forward/loss timing."""
    if n_rows < 1 or batch_size < 1 or warmup_batches < 0 or measured_batches < 1:
        raise ValueError("n_rows, batch_size, and measured_batches must be positive")

    device = resolve_device(device_name)
    io_start = time.perf_counter()
    frame = load_nidd_csv(dataset_path, n_rows=n_rows)
    io_seconds = time.perf_counter() - io_start
    model, preprocessor, checkpoint = load_trained_model(checkpoint_path, device)

    preprocess_start = time.perf_counter()
    batch = preprocessor.transform_with_mask(
        frame,
        mask_ratio=checkpoint["model_config"]["mask_ratio"],
        seed=mask_seed,
    )
    preprocess_seconds = time.perf_counter() - preprocess_start
    loader = DataLoader(
        TensorDataset(batch.context, batch.numeric, batch.targets, batch.mask),
        batch_size=batch_size,
        shuffle=False,
    )
    batches = list(loader)
    if not batches:
        raise ValueError("The selected dataset produced no profiling batches")

    def run_batch(items):
        context, numeric, targets, mask = (item.to(device) for item in items)
        output = model(context, numeric)
        loss = model.reconstruction_loss(output, targets, mask)
        return int(context.shape[0]), int(mask.sum().item()), loss.item()

    warmup_times = []
    with torch.no_grad():
        for index in range(warmup_batches):
            _synchronize(device)
            start = time.perf_counter()
            run_batch(batches[index % len(batches)])
            _synchronize(device)
            warmup_times.append(time.perf_counter() - start)

        measured_times = []
        total_rows = 0
        total_targets = 0
        weighted_loss = 0.0
        for index in range(measured_batches):
            _synchronize(device)
            start = time.perf_counter()
            rows, targets, loss = run_batch(batches[index % len(batches)])
            _synchronize(device)
            measured_times.append(time.perf_counter() - start)
            total_rows += rows
            total_targets += targets
            weighted_loss += loss * targets

    model_seconds = sum(measured_times)
    result = {
        "dataset_path": str(dataset_path),
        "checkpoint_path": str(checkpoint_path),
        "device": str(device),
        "rows_loaded": len(frame),
        "rows_measured": total_rows,
        "batch_size": batch_size,
        "warmup_batches": warmup_batches,
        "measured_batches": measured_batches,
        "io_seconds": io_seconds,
        "preprocessing_seconds": preprocess_seconds,
        "warmup_seconds": sum(warmup_times),
        "model_seconds": model_seconds,
        "end_to_end_seconds": io_seconds + preprocess_seconds + model_seconds,
        "model_rows_per_second": total_rows / model_seconds,
        "model_milliseconds_per_row": model_seconds / total_rows * 1000,
        "end_to_end_milliseconds_per_row": (
            (io_seconds + preprocess_seconds + model_seconds) / total_rows * 1000
        ),
        "reconstruction_loss": weighted_loss / total_targets,
        "total_masked_targets": total_targets,
    }
    Path(output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_path")
    parser.add_argument("checkpoint_path")
    parser.add_argument("output_path")
    parser.add_argument("--n-rows", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--measured-batches", type=int, default=10)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    profile_model(
        args.dataset_path,
        args.checkpoint_path,
        args.output_path,
        n_rows=args.n_rows,
        batch_size=args.batch_size,
        warmup_batches=args.warmup_batches,
        measured_batches=args.measured_batches,
        device_name=args.device,
    )
