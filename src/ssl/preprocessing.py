"""Leakage-safe preprocessing for the 5G-NIDD masked-reconstruction task."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch

try:
    from .masked_reconstruction import CONTEXT_COLUMNS, MASKED_COLUMNS
except ImportError:
    from masked_reconstruction import CONTEXT_COLUMNS, MASKED_COLUMNS

LABEL_COLUMNS = ("Label", "Attack Type", "Attack Tool")
INDEX_COLUMNS = ("Unnamed: 0", "")


@dataclass
class PreprocessedBatch:
    """Tensor inputs and targets consumed by the SSL model."""

    context: torch.Tensor
    numeric: torch.Tensor
    targets: torch.Tensor
    mask: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.numeric.shape[0])


class NIDDPreprocessor:
    """Fit and apply the Phase 1 feature contract for 5G-NIDD rows."""

    def __init__(
        self,
        context_columns: Iterable[str] = CONTEXT_COLUMNS,
        masked_columns: Iterable[str] = MASKED_COLUMNS,
    ) -> None:
        self.context_columns = tuple(context_columns)
        self.masked_columns = tuple(masked_columns)
        self._category_to_id: dict[str, dict[str, int]] = {}
        self._numeric_medians: dict[str, float] = {}
        self._numeric_means: dict[str, float] = {}
        self._numeric_stds: dict[str, float] = {}
        self._fitted = False

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return self.context_columns + self.masked_columns

    @property
    def category_sizes(self) -> dict[str, int]:
        self._require_fitted()
        return {
            column: len(vocab) + 2
            for column, vocab in self._category_to_id.items()
        }

    def fit(self, frame: pd.DataFrame) -> "NIDDPreprocessor":
        """Fit vocabularies and numeric statistics using only training rows."""
        self._validate_columns(frame)

        for column in self.context_columns:
            values = self._as_category_values(frame[column])
            categories = sorted(set(values))
            # ID 0 is reserved for missing values and ID 1 for unseen values.
            self._category_to_id[column] = {
                value: index + 2 for index, value in enumerate(categories)
            }

        numeric = frame.loc[:, self.masked_columns].apply(pd.to_numeric, errors="coerce")
        for column in self.masked_columns:
            values = numeric[column]
            median = float(values.median()) if values.notna().any() else 0.0
            filled = values.fillna(median)
            mean = float(filled.mean())
            std = float(filled.std(ddof=0))
            self._numeric_medians[column] = median
            self._numeric_means[column] = mean
            self._numeric_stds[column] = std if std > 0.0 else 1.0

        self._fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform rows into categorical IDs and normalized numeric features."""
        self._require_fitted()
        self._validate_columns(frame)

        context_values = []
        for column in self.context_columns:
            vocabulary = self._category_to_id[column]
            values = self._as_category_values(frame[column])
            context_values.append(
                [vocabulary.get(value, 1) if value else 0 for value in values]
            )

        context = torch.tensor(context_values, dtype=torch.long).transpose(0, 1)
        numeric_values = []
        for column in self.masked_columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            values = values.fillna(self._numeric_medians[column])
            normalized = (
                values - self._numeric_means[column]
            ) / self._numeric_stds[column]
            numeric_values.append(normalized.to_numpy(dtype="float32"))

        numeric_array = pd.DataFrame(numeric_values).to_numpy(dtype="float32", copy=True)
        numeric = torch.from_numpy(numeric_array).transpose(0, 1)
        return context, numeric

    def transform_with_mask(
        self,
        frame: pd.DataFrame,
        mask_ratio: float = 0.3,
        seed: int = 0,
    ) -> PreprocessedBatch:
        """Transform rows and deterministically mask numeric reconstruction targets."""
        if not 0.0 < mask_ratio <= 1.0:
            raise ValueError("mask_ratio must be greater than 0 and at most 1")

        context, targets = self.transform(frame)
        generator = torch.Generator().manual_seed(seed)
        mask = torch.rand(targets.shape, generator=generator) < mask_ratio
        if targets.shape[1] > 0:
            empty_rows = ~mask.any(dim=1)
            empty_indices = empty_rows.nonzero(as_tuple=False).flatten()
            if len(empty_indices):
                selected = torch.randint(
                    targets.shape[1], (len(empty_indices),), generator=generator
                )
                mask[empty_indices, selected] = True

        numeric = targets.masked_fill(mask, 0.0)
        return PreprocessedBatch(
            context=context,
            numeric=numeric,
            targets=targets,
            mask=mask,
        )

    def save_metadata(self, path: str | Path) -> None:
        """Save fitted preprocessing metadata in a human-readable JSON file."""
        import json

        self._require_fitted()
        metadata = {
            "context_columns": list(self.context_columns),
            "masked_columns": list(self.masked_columns),
            "category_to_id": self._category_to_id,
            "numeric_medians": self._numeric_medians,
            "numeric_means": self._numeric_means,
            "numeric_stds": self._numeric_stds,
        }
        Path(path).write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def from_metadata(cls, metadata: dict) -> "NIDDPreprocessor":
        """Restore a fitted preprocessor saved in a training checkpoint."""
        preprocessor = cls(
            context_columns=metadata["context_columns"],
            masked_columns=metadata["masked_columns"],
        )
        preprocessor._category_to_id = metadata["category_to_id"]
        preprocessor._numeric_medians = {
            column: float(value)
            for column, value in metadata["numeric_medians"].items()
        }
        preprocessor._numeric_means = {
            column: float(value)
            for column, value in metadata["numeric_means"].items()
        }
        preprocessor._numeric_stds = {
            column: float(value)
            for column, value in metadata["numeric_stds"].items()
        }
        preprocessor._fitted = True
        return preprocessor

    def _validate_columns(self, frame: pd.DataFrame) -> None:
        required = set(self.feature_columns)
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"Missing required NIDD columns: {', '.join(missing)}")

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Fit NIDDPreprocessor on training rows before transforming")

    @staticmethod
    def _as_category_values(series: pd.Series) -> list[str]:
        return ["" if pd.isna(value) else str(value) for value in series]


def split_by_sequence(
    frame: pd.DataFrame, validation_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows in sequence order, keeping later rows for validation."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be greater than 0 and less than 1")
    if frame.empty:
        raise ValueError("Cannot split an empty dataframe")

    ordered = frame.sort_values("Seq", kind="stable") if "Seq" in frame else frame.copy()
    split_index = max(1, min(len(ordered) - 1, int(len(ordered) * (1 - validation_fraction))))
    return ordered.iloc[:split_index].copy(), ordered.iloc[split_index:].copy()


def load_nidd_csv(path: str | Path, n_rows: int | None = None) -> pd.DataFrame:
    """Load NIDD rows and remove the exported CSV index column."""
    frame = pd.read_csv(path, nrows=n_rows)
    removable = [column for column in INDEX_COLUMNS if column in frame.columns]
    return frame.drop(columns=removable)


__all__ = [
    "LABEL_COLUMNS",
    "NIDDPreprocessor",
    "PreprocessedBatch",
    "load_nidd_csv",
    "split_by_sequence",
]
