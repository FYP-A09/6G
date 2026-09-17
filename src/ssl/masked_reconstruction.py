"""
Masked-reconstruction SSL pretext task (Sriranjana C's task — see
docs/team/sriranjana_c/design.md and dataset_notes.md).

Trains a flow-level encoder on 5G-NIDD by masking the "volume" and "rate/load"
feature groups per row and reconstructing them from the remaining (always-visible)
protocol/context columns. The encoder's output is the fixed-size embedding handed
to the TGNN, per docs/architecture/interface_contracts.md §1 (D=64).

This is a design skeleton: the model/training loop shapes match the agreed
contract, but it hasn't been run end-to-end (no torch install / GPU on this
machine yet — see the requirements note at the bottom).
"""

from __future__ import annotations

from dataclasses import dataclass

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
    learning_rate: float = 1e-3


class FlowFeatureEncoder:
    """
    Flow-level encoder variant from design.md (used for 5G-NIDD, which has no
    inherent graph — contrast with the hierarchical GNN encoder variant that
    Thrishala's ego-network-based sources would use).

    Architecture (to implement once torch is available):
        input: [context_features (categorical, embedded) ++ masked_features (zeroed)]
        -> MLP encoder -> embedding (D=64)
        -> MLP decoder -> reconstructed masked_features
    Loss: MSE between reconstructed and true (unmasked) values, on the masked
    positions only.
    """

    def __init__(self, config: MaskedReconstructionConfig = MaskedReconstructionConfig()):
        self.config = config
        # TODO once torch is installed:
        #   self.context_embedder = nn.ModuleDict({col: nn.Embedding(...) for col in CONTEXT_COLUMNS})
        #   self.encoder = nn.Sequential(nn.Linear(input_dim, config.hidden_dim), nn.ReLU(),
        #                                 nn.Linear(config.hidden_dim, config.embedding_dim))
        #   self.decoder = nn.Sequential(nn.Linear(config.embedding_dim, config.hidden_dim), nn.ReLU(),
        #                                 nn.Linear(config.hidden_dim, len(MASKED_COLUMNS)))

    def apply_mask(self, row: dict[str, float]) -> tuple[dict, dict]:
        """Returns (masked_row, targets) — targets are the true values at masked positions."""
        import random

        targets = {}
        masked_row = dict(row)
        n_to_mask = max(1, int(len(MASKED_COLUMNS) * self.config.mask_ratio))
        for col in random.sample(MASKED_COLUMNS, n_to_mask):
            targets[col] = row[col]
            masked_row[col] = 0.0
        return masked_row, targets

    def encode(self, row: dict[str, float]):
        """Forward pass -> embedding. Placeholder until the torch model exists."""
        raise NotImplementedError("Wire up self.encoder once torch/pandas are installed.")

    def evaluate_reconstruction_loss(self, dataset_path: str, n_rows: int = 10_000) -> float:
        """
        Measures reconstruction MSE over a sample of rows, and wall-clock time —
        addresses NFR1 in requirements.md ("document the computational overhead").
        """
        import csv
        import time

        start = time.time()
        total_rows = 0
        with open(dataset_path, encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for i, _row in enumerate(reader):
                if i >= n_rows:
                    break
                total_rows += 1
                # masked_row, targets = self.apply_mask(_row)
                # embedding = self.encode(masked_row)
                # reconstructed = self.decoder(embedding)  # once implemented
        elapsed = time.time() - start
        print(f"Profiled {total_rows} rows in {elapsed:.2f}s "
              f"({elapsed / max(total_rows, 1) * 1000:.3f} ms/row) — "
              f"model forward pass not yet wired up, this is I/O-only timing.")
        return elapsed


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

# Requirements to actually train this: torch (or tensorflow), pandas.
# Neither is installed on this machine yet — that install is the next concrete step.
