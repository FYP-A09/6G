"""Milan sample adapter for SSL node/timestep representations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch

MILAN_COLUMNS = (
    "CellID",
    "datetime",
    "countrycode",
    "smsin",
    "smsout",
    "callin",
    "callout",
    "internet",
)
TRAFFIC_COLUMNS = ("sms_in", "sms_out", "call_in", "call_out", "internet")
RAW_TRAFFIC_COLUMNS = {
    "smsin": "sms_in",
    "smsout": "sms_out",
    "callin": "call_in",
    "callout": "call_out",
    "internet": "internet",
}


@dataclass
class MilanBatch:
    """Normalized Milan traffic rows and their SSL/TGNN metadata."""

    features: torch.Tensor
    node_ids: list[str]
    timestamps: torch.Tensor
    source_dataset: str = "milan"

    @property
    def batch_size(self) -> int:
        return int(self.features.shape[0])


class MilanPreprocessor:
    """Aggregate and normalize Milan traffic using training data statistics."""

    def __init__(self, traffic_columns: Iterable[str] = TRAFFIC_COLUMNS) -> None:
        self.traffic_columns = tuple(traffic_columns)
        self._means: dict[str, float] = {}
        self._stds: dict[str, float] = {}
        self._fitted = False

    def fit(self, frame: pd.DataFrame) -> "MilanPreprocessor":
        self._validate_frame(frame)
        for column in self.traffic_columns:
            values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
            self._means[column] = float(values.mean())
            std = float(values.std(ddof=0))
            self._stds[column] = std if std > 0.0 else 1.0
        self._fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> MilanBatch:
        if not self._fitted:
            raise RuntimeError("Fit MilanPreprocessor on training rows first")
        self._validate_frame(frame)
        values = []
        for column in self.traffic_columns:
            numeric = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
            values.append(
                ((numeric - self._means[column]) / self._stds[column])
                .to_numpy(dtype="float32")
            )
        feature_array = pd.DataFrame(values).to_numpy(dtype="float32", copy=True).T
        return MilanBatch(
            features=torch.from_numpy(feature_array),
            node_ids=frame["CellID"].astype(str).tolist(),
            timestamps=torch.tensor(
                pd.to_numeric(frame["datetime"], errors="raise").to_numpy(dtype="int64"),
                dtype=torch.int64,
            ),
        )

    def save_metadata(self, path: str | Path) -> None:
        import json

        if not self._fitted:
            raise RuntimeError("Fit MilanPreprocessor before saving metadata")
        metadata = {
            "traffic_columns": list(self.traffic_columns),
            "means": self._means,
            "stds": self._stds,
            "source_dataset": "milan",
        }
        Path(path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def _validate_frame(self, frame: pd.DataFrame) -> None:
        required = {"CellID", "datetime", *self.traffic_columns}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"Missing Milan columns: {', '.join(missing)}")


def aggregate_milan_csv(path: str | Path) -> pd.DataFrame:
    """Aggregate country rows into one traffic vector per cell and timestamp."""
    frame = pd.read_csv(path, usecols=list(MILAN_COLUMNS))
    renamed = frame.rename(columns=RAW_TRAFFIC_COLUMNS)
    for column in TRAFFIC_COLUMNS:
        renamed[column] = pd.to_numeric(renamed[column], errors="coerce").fillna(0.0)
    grouped = (
        renamed.groupby(["CellID", "datetime"], as_index=False, sort=True)[list(TRAFFIC_COLUMNS)]
        .sum()
        .sort_values(["datetime", "CellID"], kind="stable")
        .reset_index(drop=True)
    )
    grouped["CellID"] = grouped["CellID"].astype(str)
    grouped["datetime"] = pd.to_numeric(grouped["datetime"], errors="raise").astype("int64")
    return grouped


def load_milan_sample(data_dir: str | Path) -> pd.DataFrame:
    """Aggregate all daily Milan CSV files in chronological filename order."""
    directory = Path(data_dir)
    files = sorted(directory.glob("sms-call-internet-mi-*.csv"))
    if not files:
        raise FileNotFoundError(f"No Milan sample CSV files found under {directory}")
    frames = [aggregate_milan_csv(path) for path in files]
    return pd.concat(frames, ignore_index=True).sort_values(
        ["datetime", "CellID"], kind="stable"
    ).reset_index(drop=True)


def split_milan_by_time(
    frame: pd.DataFrame, validation_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep the latest chronological intervals for validation."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be greater than 0 and less than 1")
    if frame.empty:
        raise ValueError("Cannot split an empty Milan frame")
    ordered = frame.sort_values(["datetime", "CellID"], kind="stable")
    timestamps = ordered["datetime"].drop_duplicates().tolist()
    if len(timestamps) < 2:
        raise ValueError(
            "Cannot split a Milan frame with fewer than 2 distinct timestamps "
            f"(got {len(timestamps)}) into train/validation"
        )
    split_index = max(1, min(len(timestamps) - 1, int(len(timestamps) * (1 - validation_fraction))))
    cutoff = timestamps[split_index]
    train = ordered[ordered["datetime"] < cutoff].copy()
    validation = ordered[ordered["datetime"] >= cutoff].copy()
    return train.reset_index(drop=True), validation.reset_index(drop=True)


__all__ = [
    "MilanBatch",
    "MilanPreprocessor",
    "aggregate_milan_csv",
    "load_milan_sample",
    "split_milan_by_time",
]
