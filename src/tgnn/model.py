"""
GraphSAGE + temporal-GRU TGNN (Thrishala S N's task — see design.md).

Uses the official PyTorch Geometric `SAGEConv` for spatial aggregation. Temporal
propagation uses a plain `nn.GRUCell` applied per-node over the sequence of
spatially-aggregated states, rather than PyTorch Geometric Temporal's `GConvGRU`:
that package's top-level `__init__.py` unconditionally imports every recurrent
model it ships (including unrelated ones like EvolveGCNH), and that import chain
hard-requires `torch_sparse`, which fails to compile on this project's Windows
dev machine (MSVC/PyTorch C++ ABI mismatch — confirmed, not a version/flag issue).
A GRUCell over SAGEConv's per-timestep output plays the same architectural role
(temporal state carried alongside spatially-aggregated features) without that
dependency. Revisit GConvGRU if the team moves to a Linux/CUDA environment where
torch_sparse installs cleanly.

Runnable against the real interface contract
(docs/architecture/interface_contracts.md): input is Sriranjana's D=64 SSL
embedding per node per timestep, output is Krish's predicted-demand vector.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

SSL_EMBEDDING_DIM = 64  # matches interface_contracts.md §1


class GraphSAGELayer(nn.Module):
    """
    Official PyTorch Geometric mean-aggregation GraphSAGE layer. The public
    model API remains dense-adjacency based, so this wrapper converts each
    timestep's dense structural adjacency to PyG's ``edge_index`` format.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv = SAGEConv(in_dim, out_dim, aggr="mean")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x:          [N, in_dim]  node features
        edge_index: [2, E]       PyG edge index (already converted from adjacency
                    by the caller — see TGNNPredictor.forward, which converts
                    once per distinct adjacency instead of once per layer)
        returns: [N, out_dim]
        """
        return F.relu(self.conv(x, edge_index))


class TGNNPredictor(nn.Module):
    """
    Full model: SSL embeddings [N, T, D] + adjacency [N, N]
             -> spatial GraphSAGE aggregation per timestep
             -> per-node GRU propagation over the resulting sequence
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
        self.temporal_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.output_head = nn.Linear(hidden_dim, len(self.OUTPUT_FIELDS))

    @staticmethod
    def _to_edge_index(adj: torch.Tensor) -> torch.Tensor:
        return adj.nonzero(as_tuple=False).transpose(0, 1).contiguous()

    def forward(self, embeddings: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        embeddings: [N, T, D]  SSL embeddings per node per timestep
        adj:        [N, N] or [T, N, N] graph adjacency. A 2-D adjacency is
                reused at every timestep; a 3-D adjacency captures handovers.
        returns:    [N, len(OUTPUT_FIELDS)]  prediction for the next timestep
        """
        n, t, d = embeddings.shape
        if adj.ndim == 2:
            if adj.shape != (n, n):
                raise ValueError(f"static adj must have shape [{n}, {n}]")
            # Static adjacency: convert to edge_index once, reuse at every
            # timestep and every sage layer instead of recomputing an O(N^2)
            # nonzero() scan (num_sage_layers + 1) * T times.
            static_edge_index = self._to_edge_index(adj)
            edge_index_per_step = [static_edge_index] * t
        elif adj.ndim == 3:
            if adj.shape != (t, n, n):
                raise ValueError(f"dynamic adj must have shape [{t}, {n}, {n}]")
            edge_index_per_step = [self._to_edge_index(adj[ts]) for ts in range(t)]
        else:
            raise ValueError("adj must have shape [N, N] or [T, N, N]")

        spatial_out = embeddings
        for layer in self.sage_layers:
            # Apply the same spatial layer independently at each timestep.
            spatial_out = torch.stack(
                [layer(spatial_out[:, ts, :], edge_index_per_step[ts]) for ts in range(t)], dim=1
            )  # [N, T, hidden_dim]

        hidden_state = torch.zeros(n, spatial_out.shape[-1], device=embeddings.device)
        for ts in range(t):
            hidden_state = self.temporal_cell(spatial_out[:, ts, :], hidden_state)
        last_step = hidden_state  # most recent GRU state, used to predict next
        raw_prediction = self.output_head(last_step)
        nonnegative = F.softplus(raw_prediction[:, :2])
        normalized_load = torch.sigmoid(raw_prediction[:, 2:3])
        return torch.cat([nonnegative, normalized_load], dim=-1)


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

    # Also confirm the dynamic (per-timestep) adjacency path still works.
    dummy_dynamic_adj = (torch.rand(n_timesteps, n_nodes, n_nodes) > 0.9).float()
    dynamic_prediction = model(dummy_embeddings, dummy_dynamic_adj)
    print(f"Dynamic-adjacency output: {tuple(dynamic_prediction.shape)}")
