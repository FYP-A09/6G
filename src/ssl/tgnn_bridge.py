"""Bridge exported SSL embeddings into the TGNN [N, T, D] contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

EMBEDDING_DIM = 64


@dataclass
class TGNNSslWindow:
    """A validated embedding window and its ordering metadata."""

    embeddings: torch.Tensor
    node_ids: list[str]
    timestamps: list[float]
    source_dataset: str

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.embeddings.shape)


def load_embedding_export(path: str | Path) -> pd.DataFrame:
    """Load and validate an SSL export before reshaping it for the TGNN."""
    frame = pd.read_csv(path)
    embedding_columns = sorted(
        [column for column in frame.columns if column.startswith("embedding_")]
    )
    required = {"node_id", "timestamp", "source_dataset"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing embedding metadata columns: {', '.join(missing)}")
    if len(embedding_columns) != EMBEDDING_DIM:
        raise ValueError(
            f"Expected {EMBEDDING_DIM} embedding columns, found {len(embedding_columns)}"
        )
    if frame.duplicated(["node_id", "timestamp"]).any():
        raise ValueError("Embedding export contains duplicate node/timestamp rows")
    sources = frame["source_dataset"].dropna().astype(str).unique().tolist()
    if len(sources) != 1:
        raise ValueError("Embedding export must contain exactly one source_dataset")
    return frame


def build_tgnn_window(
    path: str | Path,
    node_ids: list[str] | None = None,
    timestamps: list[float] | None = None,
) -> TGNNSslWindow:
    """Return embeddings ordered as [nodes, timestamps, 64]."""
    frame = load_embedding_export(path)
    frame["node_id"] = frame["node_id"].astype(str)
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="raise").astype("float64")
    ordered_nodes = node_ids or sorted(frame["node_id"].unique().tolist())
    ordered_timestamps = timestamps or sorted(frame["timestamp"].unique().tolist())
    selected = frame[
        frame["node_id"].isin(ordered_nodes) & frame["timestamp"].isin(ordered_timestamps)
    ]
    expected = len(ordered_nodes) * len(ordered_timestamps)
    if len(selected) != expected:
        raise ValueError(
            f"Incomplete embedding window: expected {expected} rows, found {len(selected)}"
        )
    selected = selected.set_index(["node_id", "timestamp"]).reindex(
        pd.MultiIndex.from_product(
            [ordered_nodes, ordered_timestamps], names=["node_id", "timestamp"]
        )
    )
    if selected.isna().any().any():
        raise ValueError("Embedding window contains missing node/timestamp values")
    embedding_columns = [f"embedding_{index:02d}" for index in range(EMBEDDING_DIM)]
    embeddings = torch.tensor(
        selected[embedding_columns].to_numpy(dtype="float32").reshape(
            len(ordered_nodes), len(ordered_timestamps), EMBEDDING_DIM
        ),
        dtype=torch.float32,
    )
    return TGNNSslWindow(
        embeddings=embeddings,
        node_ids=ordered_nodes,
        timestamps=ordered_timestamps,
        source_dataset=str(frame["source_dataset"].iloc[0]),
    )


__all__ = ["TGNNSslWindow", "build_tgnn_window", "load_embedding_export"]
