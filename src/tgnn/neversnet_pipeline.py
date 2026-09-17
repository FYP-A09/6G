"""Prepare NeversNet5G SSL windows and dynamic graph inputs for the TGNN."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import networkx as nx
import torch

try:
    from ..ssl.tgnn_bridge import EMBEDDING_DIM
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ssl"))
    from tgnn_bridge import EMBEDDING_DIM


def load_embedding_rows(path: str | Path) -> dict[tuple[str, float], torch.Tensor]:
    """Load NeversNet5G embeddings keyed by exact node and float timestamp."""
    embedding_columns = [f"embedding_{index:02d}" for index in range(EMBEDDING_DIM)]
    rows: dict[tuple[str, float], torch.Tensor] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"node_id", "timestamp", "source_dataset", *embedding_columns}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError("Embedding export does not match the 64-D TGNN contract")
        for row in reader:
            if row["source_dataset"] != "neversnet5g":
                raise ValueError("Expected source_dataset=neversnet5g")
            key = (row["node_id"], float(row["timestamp"]))
            if key in rows:
                raise ValueError(f"Duplicate embedding row: {key}")
            rows[key] = torch.tensor(
                [float(row[column]) for column in embedding_columns], dtype=torch.float32
            )
    return rows


def load_snapshot_window(
    snapshot_dir: str | Path, snapshot_indices: list[int]
) -> tuple[list[float], list[nx.Graph]]:
    """Load saved graph snapshots and their manifest timestamps."""
    directory = Path(snapshot_dir)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    starts = manifest["bin_starts_s"]
    timestamps, snapshots = [], []
    for index in snapshot_indices:
        if index < 0 or index >= len(starts):
            raise IndexError(f"Snapshot index out of range: {index}")
        timestamps.append(float(starts[index]))
        snapshots.append(nx.read_gml(directory / f"snapshot_{index:05d}.gml"))
    return timestamps, snapshots


def build_neversnet_window(
    embedding_path: str | Path,
    snapshot_dir: str | Path,
    snapshot_indices: list[int],
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[float]]:
    """Return embeddings ``[N,T,64]`` and dynamic adjacency ``[T,N,N]``.

    The current SSL export contains UE embeddings. For this first TGNN path, a
    gNB embedding is the mean of embeddings of its connected UEs at that step.
    """
    timestamps, snapshots = load_snapshot_window(snapshot_dir, snapshot_indices)
    embeddings = load_embedding_rows(embedding_path)
    ue_nodes = sorted(
        node
        for graph in snapshots
        for node, data in graph.nodes(data=True)
        if data.get("type") == "ue"
        and all((node, timestamp) in embeddings for timestamp in timestamps)
    )
    gnb_nodes = sorted(
        node
        for graph in snapshots
        for node, data in graph.nodes(data=True)
        if data.get("type") == "gnb"
    )
    if not ue_nodes:
        raise ValueError("No UE has a complete embedding window for these snapshots")

    node_ids = ue_nodes + gnb_nodes
    node_index = {node: index for index, node in enumerate(node_ids)}
    feature_tensor = torch.zeros(len(node_ids), len(timestamps), EMBEDDING_DIM)
    adjacency = torch.zeros(len(timestamps), len(node_ids), len(node_ids))

    for time_index, (timestamp, graph) in enumerate(zip(timestamps, snapshots)):
        for node in ue_nodes:
            feature_tensor[node_index[node], time_index] = embeddings[(node, timestamp)]
        for gnb in gnb_nodes:
            neighbors = [
                node for node in graph.neighbors(gnb)
                if node in node_index and node in ue_nodes
            ]
            if neighbors:
                feature_tensor[node_index[gnb], time_index] = torch.stack(
                    [embeddings[(node, timestamp)] for node in neighbors]
                ).mean(dim=0)
        for source, target in graph.edges:
            if source in node_index and target in node_index:
                adjacency[time_index, node_index[source], node_index[target]] = 1.0
                adjacency[time_index, node_index[target], node_index[source]] = 1.0

    return feature_tensor, adjacency, node_ids, timestamps


__all__ = ["build_neversnet_window", "load_embedding_rows", "load_snapshot_window"]
