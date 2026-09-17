"""
Masked-reconstruction SSL pretext task (Sriranjana C's task — see
docs/team/sriranjana_c/design.md and dataset_notes.md).

Trains a flow-level encoder on 5G-NIDD by masking the "volume" and "rate/load"
feature groups per row and reconstructing them from the remaining (always-visible)
protocol/context columns. The encoder's output is the fixed-size embedding handed
to the TGNN, per docs/architecture/interface_contracts.md §1 (D=64).

The Phase 1 preprocessing contract feeds the PyTorch encoder and decoder here.
The training loop remains a Phase 3 task.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

# Columns from docs/team/sriranjana_c/dataset_notes.md
CONTEXT_COLUMNS = [
    "Proto", "sTos", "dTos", "sDSb", "dDSb", "sTtl", "dTtl",
    "sHops", "dHops", "Cause", "State",
]
MASKED_COLUMNS = [
    # Volume
    "TotPkts", "SrcPkts", "DstPkts", "TotBytes", "SrcBytes", "DstBytes",
    "sMeanPktSz", "dMeanPktSz",
    # Rate/load
    "Load", "SrcLoad", "DstLoad", "Rate", "SrcRate", "DstRate",
]

EMBEDDING_DIM = 64  # matches docs/architecture/interface_contracts.md §1


@dataclass
class MaskedReconstructionConfig:
    mask_ratio: float = 0.3  # fraction of MASKED_COLUMNS zeroed out per row
    embedding_dim: int = EMBEDDING_DIM
    hidden_dim: int = 128
    categorical_embedding_dim: int = 8
    learning_rate: float = 1e-3


@dataclass
class MaskedReconstructionOutput:
    embedding: torch.Tensor
    reconstruction: torch.Tensor


class FlowFeatureEncoder(nn.Module):
    """
    Flow-level encoder variant from design.md (used for 5G-NIDD, which has no
    inherent graph — contrast with the hierarchical GNN encoder variant that
    Thrishala's ego-network-based sources would use).

    Architecture:
        input: [context_features (categorical, embedded) ++ masked_features (zeroed)]
        -> MLP encoder -> embedding (D=64)
        -> MLP decoder -> reconstructed masked_features
    Loss: MSE between reconstructed and true (unmasked) values, on the masked
    positions only.
    """

    def __init__(
        self,
        config: MaskedReconstructionConfig | None = None,
        category_sizes: dict[str, int] | None = None,
    ):
        super().__init__()
        config = config or MaskedReconstructionConfig()
        self.config = config
        category_sizes = category_sizes or {column: 2 for column in CONTEXT_COLUMNS}
        missing_sizes = set(CONTEXT_COLUMNS).difference(category_sizes)
        if missing_sizes:
            raise ValueError(
                f"Missing category sizes for: {', '.join(sorted(missing_sizes))}"
            )
        if any(size < 2 for size in category_sizes.values()):
            raise ValueError("Each category vocabulary must reserve missing and unknown IDs")

        self.context_embedder = nn.ModuleDict(
            {
                column: nn.Embedding(
                    category_sizes[column], config.categorical_embedding_dim
                )
                for column in CONTEXT_COLUMNS
            }
        )
        input_dim = (
            len(CONTEXT_COLUMNS) * config.categorical_embedding_dim
            + len(MASKED_COLUMNS)
        )
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, len(MASKED_COLUMNS)),
        )

    def forward(
        self, context: torch.Tensor, masked_numeric: torch.Tensor
    ) -> MaskedReconstructionOutput:
        """Encode context plus masked numeric features and reconstruct targets."""
        if context.ndim != 2 or context.shape[1] != len(CONTEXT_COLUMNS):
            raise ValueError(
                f"context must have shape [batch, {len(CONTEXT_COLUMNS)}]"
            )
        if masked_numeric.ndim != 2 or masked_numeric.shape[1] != len(MASKED_COLUMNS):
            raise ValueError(
                f"masked_numeric must have shape [batch, {len(MASKED_COLUMNS)}]"
            )
        if context.shape[0] != masked_numeric.shape[0]:
            raise ValueError("context and masked_numeric must have the same batch size")

        embedded_context = [
            self.context_embedder[column](context[:, index])
            for index, column in enumerate(CONTEXT_COLUMNS)
        ]
        model_input = torch.cat([*embedded_context, masked_numeric], dim=1)
        embedding = self.encoder(model_input)
        reconstruction = self.decoder(embedding)
        return MaskedReconstructionOutput(embedding, reconstruction)

    def encode(self, context: torch.Tensor, masked_numeric: torch.Tensor) -> torch.Tensor:
        """Return only the fixed-size embedding for downstream consumers."""
        return self.forward(context, masked_numeric).embedding

    @staticmethod
    def reconstruction_loss(
        output: MaskedReconstructionOutput,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate MSE only over values selected by the masking operation."""
        if targets.shape != output.reconstruction.shape or targets.shape != mask.shape:
            raise ValueError("reconstruction, targets, and mask must have identical shapes")
        selected_errors = (output.reconstruction - targets).pow(2)[mask]
        if selected_errors.numel() == 0:
            raise ValueError("mask must select at least one reconstruction target")
        return selected_errors.mean()

if __name__ == "__main__":
    import os

    encoder = FlowFeatureEncoder()
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "raw", "5g_nidd", "Combined.csv"
    )
    if os.path.exists(data_path):
        encoder.evaluate_reconstruction_loss(data_path, n_rows=1000)
    else:
        print(f"5G-NIDD not found at {data_path} — "
              f"run: python src/data/download_datasets.py kaggle")

