"""
GraphSAGE + temporal-conv TGNN (Thrishala S N's task — see design.md).

Written in plain torch (no torch_geometric — not installed on this machine yet;
`pip install torch-geometric` would let GraphSAGEConv below be replaced with
`torch_geometric.nn.SAGEConv` directly, same math, less code). Runnable today
against dummy tensors shaped like the real interface contract
(docs/architecture/interface_contracts.md): input is Sriranjana's D=64 SSL
embedding per node per timestep, output is Krish's predicted-demand vector.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

SSL_EMBEDDING_DIM = 64  # matches interface_contracts.md §1


class GraphSAGELayer(nn.Module):
    """
    Mean-aggregation GraphSAGE layer (Hamilton et al.), hand-rolled since
    torch_geometric isn't installed. Given node features and a dense adjacency
    matrix, aggregates each node's neighbor features (mean) and concatenates with
    the node's own features before a linear projection — this is exactly
    GraphSAGE's "mean aggregator" variant, the one used in the GraphSAGE-DT paper
    reviewed in the literature survey.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x:   [N, in_dim]     node features
        adj: [N, N]          binary/weighted adjacency (adj[i, j] = edge i->j)
        returns: [N, out_dim]
        """
        degree = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
        neighbor_mean = (adj @ x) / degree
        combined = torch.cat([x, neighbor_mean], dim=-1)
        return F.relu(self.linear(combined))


class TemporalConvBlock(nn.Module):
    """1D causal convolution over the time axis, per node — the SSGNN-style
    temporal component paired with GraphSAGE's spatial aggregation."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(
            channels, channels, kernel_size, padding=kernel_size - 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, T, C] -> [N, T, C] (causal: trims the right-padding)"""
        x = x.transpose(1, 2)  # [N, C, T]
        out = self.conv(x)[:, :, : x.shape[-1]]  # causal trim
        return F.relu(out).transpose(1, 2)  # back to [N, T, C]


class TGNNPredictor(nn.Module):
    """
    Full model: SSL embeddings [N, T, D] + adjacency [N, N]
             -> spatial GraphSAGE aggregation per timestep
             -> temporal conv over the resulting sequence
             -> per-node predicted-demand vector (throughput, latency, load)

    Output fields map to docs/architecture/interface_contracts.md §2
    (predicted_throughput_bps, predicted_latency_ms, predicted_load).
    """

    OUTPUT_FIELDS = ("predicted_throughput_bps", "predicted_latency_ms", "predicted_load")

    def __init__(
        self,
        embedding_dim: int = SSL_EMBEDDING_DIM,
        hidden_dim: int = 64,
        num_sage_layers: int = 2,
    ):
        super().__init__()
        dims = [embedding_dim] + [hidden_dim] * num_sage_layers
        self.sage_layers = nn.ModuleList(
            [GraphSAGELayer(dims[i], dims[i + 1]) for i in range(num_sage_layers)]
        )
        self.temporal_conv = TemporalConvBlock(hidden_dim)
        self.output_head = nn.Linear(hidden_dim, len(self.OUTPUT_FIELDS))

    def forward(self, embeddings: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        embeddings: [N, T, D]  SSL embeddings per node per timestep
        adj:        [N, N]     graph adjacency (static within the window)
        returns:    [N, len(OUTPUT_FIELDS)]  prediction for the next timestep
        """
        n, t, d = embeddings.shape
        spatial_out = embeddings
        for layer in self.sage_layers:
            # Apply the same spatial layer independently at each timestep.
            spatial_out = torch.stack(
                [layer(spatial_out[:, ts, :], adj) for ts in range(t)], dim=1
            )  # [N, T, hidden_dim]

        temporal_out = self.temporal_conv(spatial_out)  # [N, T, hidden_dim]
        last_step = temporal_out[:, -1, :]  # use the most recent step to predict next
        return self.output_head(last_step)  # [N, len(OUTPUT_FIELDS)]


if __name__ == "__main__":
    # Smoke test with dummy tensors shaped like the real contract: 29 nodes
    # (matches build_graph.py's 10-UE sample run), 5 timesteps, D=64 embeddings.
    torch.manual_seed(0)
    n_nodes, n_timesteps = 29, 5
    dummy_embeddings = torch.randn(n_nodes, n_timesteps, SSL_EMBEDDING_DIM)
    dummy_adj = (torch.rand(n_nodes, n_nodes) > 0.9).float()

    model = TGNNPredictor()
    prediction = model(dummy_embeddings, dummy_adj)
    print(f"Input:  embeddings {tuple(dummy_embeddings.shape)}, adj {tuple(dummy_adj.shape)}")
    print(f"Output: {tuple(prediction.shape)} -> fields {TGNNPredictor.OUTPUT_FIELDS}")
    print("Sample prediction (node 0):", dict(zip(TGNNPredictor.OUTPUT_FIELDS, prediction[0].tolist())))
