# TGNN Implementation Report

**Scope:** NeversNet5G graph construction, SSL-to-TGNN integration, dynamic TGNN inference, and next-step training.

## Completed

- Built regular NeversNet5G graph snapshots from event-driven UE CSV files.
- Preserved UE/gNB node identity and dynamic UE-to-gNB edges across time bins.
- Added the NeversNet5G SSL adapter and 64-dimensional export in `src/ssl/neversnet_ssl.py`.
- Verified the generated export:
  - 4,626 embedding rows
  - 171 UE node IDs
  - 40 timestamps
  - `source_dataset=neversnet5g` for every row
  - `embedding_00` through `embedding_63`
- Added `src/tgnn/neversnet_pipeline.py` to align exported UE embeddings with graph snapshot timestamps and derive temporary gNB embeddings by averaging connected UE embeddings.
- Updated `src/tgnn/model.py` to accept either static `[N,N]` or dynamic `[T,N,N]` adjacency.
- Enforced the output contract: throughput and latency are nonnegative, and load is normalized to `[0,1]`.
- Added `TARGET_FEATURE_COLUMNS` to retain populated RLC target fields in graph snapshots.
- Added `src/tgnn/train_tgnn.py` for masked next-step training with Log-Cosh loss.
- Installed PyTorch Geometric and PyTorch Geometric Temporal in the project
  `venv`; `src/tgnn/model.py` now uses official `SAGEConv` plus `GConvGRU`.

## Real execution

The real export was consumed with five graph snapshots:

```text
embeddings: [128, 5, 64]
adjacency:  [5, 128, 128]
prediction: [128, 3]
```

The real TGNN training command was:

```bash
python src/tgnn/train_tgnn.py \
  data/processed/ssl_neversnet_training/neversnet_embeddings.csv \
  data/processed/neversnet5g_graphs/part1_tgnn_targets \
  data/processed/tgnn_neversnet_training/tgnn_model.pt \
  --window-size 5 --epochs 2
```

Training completed with 4,161 masked target values:

| Epoch | Log-Cosh loss |
|---:|---:|
| 1 | 0.049281 |
| 2 | 0.003201 |

The checkpoint is written to the ignored `data/processed` directory and should not be committed.

The PyTorch Geometric Temporal model was trained with the same real sample:

```bash
/Users/thrishalasivakumar/Documents/FYP/6G/venv/bin/python src/tgnn/train_tgnn.py \
  data/processed/ssl_neversnet_training/neversnet_embeddings.csv \
  data/processed/neversnet5g_graphs/part1_tgnn_targets \
  data/processed/tgnn_neversnet_training/tgnn_pyg_temporal_model.pt \
  --window-size 5 --epochs 2
```

| Epoch | PyG Temporal Log-Cosh loss |
|---:|---:|
| 1 | 0.042433 |
| 2 | 0.003161 |

This is the active TGNN checkpoint for the current sample experiment.

## Target limitation

The sample contains no populated `throughput_dl_bps` values after resampling. It does contain `rlc_pdu_throughput_ul_bps` and `rlc_pdu_delay_ul_ms`, so the training run uses those as explicit NeversNet5G proxy targets. The public TGNN output names remain `predicted_throughput_bps`, `predicted_latency_ms`, and `predicted_load`; the target substitution must be reported in any experiment results.

## Requirements status

- **FR1:** Implemented and trained on real NeversNet5G embeddings and targets.
- **FR2:** Implemented through per-timestep graph snapshots and dynamic adjacency.
- **FR3:** B5G multi-topology loading and topology-level holdout splitting are implemented. A quantitative train-on-topologies/evaluate-on-held-out-topology experiment is still pending.
- **FR4:** Tensor output matches the TGNN contract. MARL record serialization and final slice-grain agreement remain integration work with Krish and Keerthivasan.
- **NFR1:** Dynamic graph support is present through `SAGEConv` plus `GConvGRU`; long-horizon and oversmoothing ablations are not yet measured.
- **NFR2:** Real inference executes with the PyG Temporal model; a formal latency benchmark should be recorded on the target control-loop host.

## Evidence gaps

1. NeversNet5G does not expose the true serving-cell label or per-gNB candidate SINR values, so nearest-gNB versus SINR reassignment cannot be quantitatively compared from this release. The current nearest-gNB association remains a documented heuristic.
2. PyTorch Geometric Temporal is installed in the project `venv` using
  `pip install --no-build-isolation torch-geometric-temporal`; its native
  `torch-scatter` and `torch-sparse` extensions built successfully. The package
  emits a `torch.jit.script` deprecation warning under the current PyTorch
  version, but inference and training pass.
3. The gNB embedding is currently a mean of connected UE embeddings. A future graph-aware SSL export can replace this provisional pooling step.
4. The current training run is a sample baseline, not a final full-scale result.
