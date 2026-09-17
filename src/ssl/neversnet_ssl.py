"""Train and export 64-D SSL embeddings from NeversNet5G UE telemetry.

The NeversNet5G release is an event-driven collection of per-UE CSV files.
This runner processes one UE file at a time, resamples its sparse events onto a
fixed time grid, trains a numeric masked-reconstruction autoencoder, and appends
one embedding row per UE/timestep to a CSV export.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Sequence

from networkx import config
import numpy as np
import pandas as pd
import torch
from torch import minimum, nn

try:
    from ..tgnn.build_graph import NODE_FEATURE_COLUMNS, discover_ue_files, resample_ue_to_bins
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tgnn"))
    from build_graph import NODE_FEATURE_COLUMNS, discover_ue_files, resample_ue_to_bins

LOGGER = logging.getLogger("neversnet_ssl")
SOURCE_DATASET = "neversnet5g"
EMBEDDING_DIM = 64
PART_RE = re.compile(r"(?P<part>part\d+(?:[._]5)?)")


@dataclass
class NeversNetConfig:
    """Training and resampling settings for the NeversNet5G runner."""

    bin_size_s: float = 10.0
    mask_ratio: float = 0.3
    hidden_dim: int = 128
    embedding_dim: int = EMBEDDING_DIM
    learning_rate: float = 1e-3
    epochs: int = 5
    batch_size: int = 1024
    validation_fraction: float = 0.2
    seed: int = 0
    device: str = "auto"


@dataclass
class NeversNetOutput:
    embedding: torch.Tensor
    reconstruction: torch.Tensor


class NeversNetMaskedAutoencoder(nn.Module):
    """Numeric masked-reconstruction encoder for the common UE telemetry schema."""

    def __init__(self, config: NeversNetConfig | None = None) -> None:
        super().__init__()
        config = config or NeversNetConfig()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(len(NODE_FEATURE_COLUMNS), config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, len(NODE_FEATURE_COLUMNS)),
        )

    def forward(self, features: torch.Tensor) -> NeversNetOutput:
        expected = len(NODE_FEATURE_COLUMNS)
        if features.ndim != 2 or features.shape[1] != expected:
            raise ValueError(f"features must have shape [batch, {expected}]")
        embedding = self.encoder(features)
        return NeversNetOutput(embedding, self.decoder(embedding))

    @staticmethod
    def reconstruction_loss(
        output: NeversNetOutput, targets: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        if output.reconstruction.shape != targets.shape or targets.shape != mask.shape:
            raise ValueError("reconstruction, targets, and mask must have identical shapes")
        errors = (output.reconstruction - targets).pow(2)[mask]
        if errors.numel() == 0:
            raise ValueError("mask must select at least one telemetry feature")
        return errors.mean()


@dataclass
class PartInfo:
    part_dir: Path
    part_label: str
    files: list[tuple[int, str]]
    bin_starts: list[float]
    validation_cutoff: float


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(requested)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")

    return device


def _part_label(part_dir: Path) -> str:
    match = PART_RE.search(part_dir.name)
    return match.group("part") if match else part_dir.name


def _time_bounds(csv_path: str, chunksize: int = 250_000) -> tuple[float, float] | None:
    minimum, maximum = math.inf, -math.inf
    try:
        chunks = pd.read_csv(csv_path, usecols=["time_s"], chunksize=chunksize)
        for chunk in chunks:
            values = pd.to_numeric(chunk["time_s"], errors="coerce").dropna()
            if not values.empty:
                minimum = min(minimum, float(values.min()))
                maximum = max(maximum, float(values.max()))
    except (ValueError, pd.errors.EmptyDataError):
        return None
    return None if minimum == math.inf else (minimum, maximum)


def _make_part_info(part_dir: str | Path, config: NeversNetConfig, max_files: int | None) -> PartInfo:
    directory = Path(part_dir)
    files = discover_ue_files(str(directory))
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No *_ue_*_metrics.csv files found under {directory}")

    minimum, maximum = math.inf, -math.inf
    LOGGER.info("Scanning time bounds: part=%s files=%d", directory.name, len(files))
    for index, (_, path) in enumerate(files, start=1):
        bounds = _time_bounds(path)
        if bounds:
            minimum, maximum = min(minimum, bounds[0]), max(maximum, bounds[1])
        if index == 1 or index % 25 == 0 or index == len(files):
            LOGGER.info("Time-bound scan: part=%s file=%d/%d", directory.name, index, len(files))
    if minimum == math.inf:
        raise ValueError(f"No usable time_s values found under {directory}")

    bin_count = max(2, math.ceil((maximum - minimum) / config.bin_size_s) + 1)
    bins = [minimum + index * config.bin_size_s for index in range(bin_count)]
    cutoff_index = max(1,min(len(bins) - 1,int((len(bins) - 1) * (1 - config.validation_fraction))))
    return PartInfo(directory, _part_label(directory), files, bins, bins[cutoff_index])


def _iter_file_frames(
    part: PartInfo,
    config: NeversNetConfig,
    include_validation: bool | None = None,
) -> Iterator[tuple[int, str, pd.DataFrame]]:
    """Yield one resampled UE frame at a time, never the whole part."""
    for index, (ue_id, path) in enumerate(part.files, start=1):
        started = time.perf_counter()
        try:
            frame = resample_ue_to_bins(path, part.bin_starts)
            if frame.empty:
                LOGGER.warning("Skipping empty telemetry file: %s", path)
                continue
            selected = frame.copy()
            if include_validation is True:
                selected = selected[selected["time_s"] >= part.validation_cutoff]
            elif include_validation is False:
                selected = selected[selected["time_s"] < part.validation_cutoff]
            if selected.empty:
                continue
            LOGGER.info(
                "Processed part=%s file=%d/%d ue=%s rows=%d seconds=%.2f",
                part.part_label, index, len(part.files), ue_id, len(selected),
                time.perf_counter() - started,
            )
            yield ue_id, path, selected
        except Exception:
            LOGGER.exception("Failed telemetry file; continuing: %s", path)


def _feature_frame(frame: pd.DataFrame, means: np.ndarray | None = None, stds: np.ndarray | None = None) -> np.ndarray:
    values = frame.reindex(columns=NODE_FEATURE_COLUMNS).apply(pd.to_numeric, errors="coerce")
    array = values.to_numpy(dtype="float32", copy=True)
    if means is None or stds is None:
        return array
    missing = ~np.isfinite(array)
    rows, columns = np.where(missing)
    array[rows, columns] = means[columns]
    return (array - means) / stds


def _fit_statistics(parts: Sequence[PartInfo], config: NeversNetConfig) -> tuple[np.ndarray, np.ndarray, int]:
    sums = np.zeros(len(NODE_FEATURE_COLUMNS), dtype=np.float64)
    squares = np.zeros(len(NODE_FEATURE_COLUMNS), dtype=np.float64)
    counts = np.zeros(len(NODE_FEATURE_COLUMNS), dtype=np.int64)
    LOGGER.info("Pass 1/3: fitting telemetry normalization statistics")
    for part in parts:
        for _, _, frame in _iter_file_frames(part, config, include_validation=False):
            values = _feature_frame(frame)
            valid = np.isfinite(values)
            sums += np.where(valid, values, 0.0).sum(axis=0)
            squares += np.where(valid, values * values, 0.0).sum(axis=0)
            counts += valid.sum(axis=0)
    if not counts.any():
        raise ValueError("No finite NeversNet5G feature values found")
    means = sums / np.maximum(counts, 1)
    variances = np.maximum(squares / np.maximum(counts, 1) - means * means, 0.0)
    stds = np.sqrt(variances)
    stds[stds == 0.0] = 1.0
    LOGGER.info("Fitted normalization statistics from %d feature rows", int(counts.max()))
    return means.astype("float32"), stds.astype("float32"), int(counts.max())


def _masked_batch(features: np.ndarray, mask_ratio: float, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = torch.from_numpy(np.array(features, dtype="float32", copy=True))
    generator = torch.Generator().manual_seed(seed)
    mask = torch.rand(values.shape, generator=generator) < mask_ratio
    empty = ~mask.any(dim=1)
    if empty.any():
        rows = empty.nonzero(as_tuple=False).flatten()
        columns = torch.randint(values.shape[1], (len(rows),), generator=generator)
        mask[rows, columns] = True
    return values.masked_fill(mask, 0.0), values, mask


def _run_frame(
    model: NeversNetMaskedAutoencoder,
    frame: pd.DataFrame,
    means: np.ndarray,
    stds: np.ndarray,
    config: NeversNetConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    seed: int,
) -> tuple[float, int, int]:
    features = _feature_frame(frame, means, stds)
    valid_rows = np.isfinite(features).all(axis=1)
    features = features[valid_rows]
    if len(features) == 0:
        return 0.0, 0, 0
    total_loss, total_targets = 0.0, 0
    model.train(optimizer is not None)
    for start in range(0, len(features), config.batch_size):
        masked, targets, mask = _masked_batch(
            features[start:start + config.batch_size], config.mask_ratio, seed + start
        )
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
        total_targets += count
    return total_loss, total_targets, len(features)


def _run_pass(
    parts: Sequence[PartInfo], config: NeversNetConfig, means: np.ndarray, stds: np.ndarray,
    model: NeversNetMaskedAutoencoder, device: torch.device,
    optimizer: torch.optim.Optimizer | None, include_validation: bool, seed: int,
) -> tuple[float, int, int]:
    loss, targets, rows = 0.0, 0, 0
    for part in parts:
        for file_index, (_, _, frame) in enumerate(_iter_file_frames(part, config, include_validation=include_validation)):
            current_loss, current_targets, current_rows = _run_frame(
                model, frame, means, stds, config, device, optimizer,
                seed + file_index,
            )
            loss += current_loss
            targets += current_targets
            rows += current_rows
    if targets == 0:
        raise ValueError("No masked telemetry targets were produced")
    return loss / targets, targets, rows


def _checkpoint_payload(
    model: NeversNetMaskedAutoencoder, config: NeversNetConfig,
    means: np.ndarray, stds: np.ndarray, parts: Sequence[PartInfo], best_loss: float,
) -> dict:
    return {
        "model_state_dict": model.state_dict(),
        "model_config": asdict(config),
        "feature_columns": list(NODE_FEATURE_COLUMNS),
        "normalization_means": means.tolist(),
        "normalization_stds": stds.tolist(),
        "part_labels": [part.part_label for part in parts],
        "source_dataset": SOURCE_DATASET,
        "best_validation_loss": best_loss,
    }


def train_neversnet5g(
    part_dirs: Sequence[str | Path], output_dir: str | Path,
    config: NeversNetConfig | None = None, max_files: int | None = None,
) -> list[dict]:
    """Train from one or more NeversNet5G parts using sequential file passes."""
    config = config or NeversNetConfig()
    if not 0.0 < config.mask_ratio <= 1.0:
        raise ValueError("mask_ratio must be greater than 0 and at most 1")
    _set_seed(config.seed)
    device = _resolve_device(config.device)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    parts = [_make_part_info(path, config, max_files) for path in part_dirs]
    means, stds, _ = _fit_statistics(parts, config)
    model = NeversNetMaskedAutoencoder(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    best_loss = float("inf")
    history = []
    for epoch in range(1, config.epochs + 1):
        started = time.perf_counter()
        LOGGER.info("Pass 2/3: training epoch %d/%d", epoch, config.epochs)
        train_loss, _, train_rows = _run_pass(
            parts, config, means, stds, model, device, optimizer, False, config.seed + epoch
        )
        with torch.no_grad():
            LOGGER.info("Validation pass: epoch %d/%d", epoch, config.epochs)
            validation_loss, _, validation_rows = _run_pass(
                parts, config, means, stds, model, device, None, True, config.seed + 10_000 + epoch
            )
        record = {
            "epoch": epoch, "train_loss": train_loss,
            "validation_loss": validation_loss, "train_rows": train_rows,
            "validation_rows": validation_rows, "seconds": time.perf_counter() - started,
        }
        history.append(record)
        LOGGER.info(
            "epoch=%d train_loss=%.6f validation_loss=%.6f train_rows=%d validation_rows=%d seconds=%.2f",
            epoch, train_loss, validation_loss, train_rows, validation_rows, record["seconds"],
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            torch.save(
                _checkpoint_payload(model, config, means, stds, parts, best_loss),
                output / "best_neversnet_model.pt",
            )
            LOGGER.info("Saved checkpoint: %s", output / "best_neversnet_model.pt")
    (output / "metrics.json").write_text(
        json.dumps({"source_dataset": SOURCE_DATASET, "feature_columns": list(NODE_FEATURE_COLUMNS), "best_validation_loss": best_loss, "epochs": history}, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("Training complete: best_validation_loss=%.6f", best_loss)
    return history


def export_neversnet5g_embeddings(
    part_dirs: Sequence[str | Path], checkpoint_path: str | Path,
    output_path: str | Path, max_files: int | None = None, device_name: str = "cpu",
) -> int:
    """Load a trained checkpoint and append embeddings one UE file at a time."""
    device = _resolve_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = NeversNetConfig(**checkpoint["model_config"])
    model = NeversNetMaskedAutoencoder(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    means = np.asarray(checkpoint["normalization_means"], dtype="float32")
    stds = np.asarray(checkpoint["normalization_stds"], dtype="float32")
    parts = [_make_part_info(path, config, max_files) for path in part_dirs]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    embedding_columns = [f"embedding_{index:02d}" for index in range(config.embedding_dim)]
    columns = ["node_id", "timestamp", "source_dataset", *embedding_columns]
    rows_written = 0
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for part in parts:
            LOGGER.info("Exporting part=%s files=%d", part.part_label, len(part.files))
            for file_index, (ue_id, path, frame) in enumerate(
                _iter_file_frames(part, config, include_validation=None), start=1
            ):
                features = _feature_frame(frame, means, stds)
                valid_rows = np.isfinite(features).all(axis=1)
                if not valid_rows.any():
                    LOGGER.warning("No usable feature rows for export: %s", path)
                    continue
                with torch.no_grad():
                    embedding = model(torch.from_numpy(features[valid_rows]).to(device)).embedding.cpu().numpy()
                valid_frame = frame.loc[valid_rows].reset_index(drop=True)
                node_id = f"ue_{part.part_label}_{ue_id}"
                for row_index, values in enumerate(embedding):
                    record = {
                        "node_id": node_id,
                        "timestamp": float(valid_frame.iloc[row_index]["time_s"]),
                        "source_dataset": SOURCE_DATASET,
                    }
                    record.update({column: float(value) for column, value in zip(embedding_columns, values)})
                    writer.writerow(record)
                    rows_written += 1
                LOGGER.info(
                    "Exported part=%s file=%d/%d ue=%s rows=%d total_rows=%d",
                    part.part_label, file_index, len(part.files), ue_id, len(embedding), rows_written,
                )
    LOGGER.info("Embedding export complete: rows=%d path=%s", rows_written, output)
    return rows_written


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--part-dir", action="append", required=True, help="Repeat for part1, part1_5, part2, ...")
    train.add_argument("--output-dir", required=True)
    train.add_argument("--bin-size-s", type=float, default=10.0)
    train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--batch-size", type=int, default=1024)
    train.add_argument("--max-files", type=int, default=None)
    train.add_argument("--device", default="auto")
    train.add_argument("--verbose", action="store_true")
    export = subparsers.add_parser("export")
    export.add_argument("--part-dir", action="append", required=True)
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--max-files", type=int, default=None)
    export.add_argument("--device", default="cpu")
    export.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    _configure_logging(args.verbose)
    if args.command == "train":
        train_neversnet5g(
            args.part_dir,
            args.output_dir,
            NeversNetConfig(
                bin_size_s=args.bin_size_s, epochs=args.epochs,
                batch_size=args.batch_size, device=args.device,
            ),
            max_files=args.max_files,
        )
    else:
        export_neversnet5g_embeddings(
            args.part_dir, args.checkpoint, args.output,
            max_files=args.max_files, device_name=args.device,
        )


if __name__ == "__main__":
    main()
