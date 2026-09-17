"""Train and evaluate the NeversNet5G TGNN on real next-step proxy targets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import networkx as nx
import torch
import torch.nn.functional as F

try:
    from .model import TGNNPredictor
    from .neversnet_pipeline import build_neversnet_window, load_snapshot_window
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model import TGNNPredictor
    from neversnet_pipeline import build_neversnet_window, load_snapshot_window

TARGET_FIELDS = ("rlc_pdu_throughput_ul_bps", "rlc_pdu_delay_ul_ms")


def _target_for_next_snapshot(
    graph: nx.Graph, node_ids: list[str], throughput_scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    target = torch.zeros(len(node_ids), 3, dtype=torch.float32)
    mask = torch.zeros(len(node_ids), 3, dtype=torch.bool)
    for index, node_id in enumerate(node_ids):
        data = graph.nodes[node_id] if node_id in graph else {}
        throughput = data.get(TARGET_FIELDS[0])
        latency = data.get(TARGET_FIELDS[1])
        if throughput is not None and latency is not None:
            throughput = max(float(throughput), 0.0)
            latency = max(float(latency), 0.0)
            target[index, 0] = throughput
            target[index, 1] = latency
            target[index, 2] = min(throughput / throughput_scale, 1.0)
            mask[index] = True
    return target, mask


def _log_cosh(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    error = (prediction - target).abs()
    return F.softplus(2.0 * error) - error - torch.log(torch.tensor(2.0))


def train_tgnn(
    embedding_path: str | Path,
    snapshot_dir: str | Path,
    output_path: str | Path,
    window_size: int = 5,
    epochs: int = 5,
    learning_rate: float = 1e-3,
) -> dict:
    """Train on consecutive snapshot windows with masked real proxy targets."""
    directory = Path(snapshot_dir)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    snapshot_count = int(manifest["n_snapshots"])
    if snapshot_count <= window_size:
        raise ValueError("Need more snapshots than the requested input window")

    _, all_snapshots = load_snapshot_window(directory, list(range(snapshot_count)))
    values = [
        float(data[TARGET_FIELDS[0]])
        for graph in all_snapshots
        for _, data in graph.nodes(data=True)
        if data.get(TARGET_FIELDS[0]) is not None
    ]
    if not values:
        raise ValueError("No populated NeversNet5G proxy throughput targets found")
    throughput_scale = max(max(values), 1.0)

    model = TGNNPredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        target_count = 0
        for start in range(snapshot_count - window_size):
            indices = list(range(start, start + window_size))
            embeddings, adjacency, node_ids, _ = build_neversnet_window(
                embedding_path, directory, indices
            )
            target, mask = _target_for_next_snapshot(
                all_snapshots[start + window_size], node_ids, throughput_scale
            )
            if not mask.any():
                continue
            optimizer.zero_grad(set_to_none=True)
            prediction = model(embeddings, adjacency)
            transformed_prediction = torch.stack(
                [torch.log1p(prediction[:, 0]), torch.log1p(prediction[:, 1]), prediction[:, 2]], dim=1
            )
            transformed_target = torch.stack(
                [torch.log1p(target[:, 0]), torch.log1p(target[:, 1]), target[:, 2]], dim=1
            )
            loss = _log_cosh(transformed_prediction[mask], transformed_target[mask]).mean()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(mask.sum())
            target_count += int(mask.sum())
        if not target_count:
            raise ValueError("No complete next-step target rows were found")
        record = {"epoch": epoch, "loss": total_loss / target_count, "target_values": target_count}
        history.append(record)
        print(record)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {"embedding_dim": 64, "window_size": window_size},
        "target_fields": list(TARGET_FIELDS),
        "throughput_scale": throughput_scale,
        "history": history,
    }, output)
    return {"snapshot_count": snapshot_count, "window_size": window_size, "throughput_scale": throughput_scale, "history": history}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("embedding_path")
    parser.add_argument("snapshot_dir")
    parser.add_argument("output_path")
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser


if __name__ == "__main__":
    args = _parser().parse_args()
    train_tgnn(
        args.embedding_path, args.snapshot_dir, args.output_path,
        window_size=args.window_size, epochs=args.epochs, learning_rate=args.learning_rate,
    )
