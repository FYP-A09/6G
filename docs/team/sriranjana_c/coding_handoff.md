# SSL Coding Handoff

**Owner:** Sriranjana C  
**Branch:** `feature-ssl`  
**Scope:** Completed SSL coding work and the remaining coding handoff items.  
**Last updated:** 17 September 2026

## Purpose

This document explains the SSL implementation completed so far. It is intended as a practical handoff for teammates working on the TGNN and Digital Twin integration.

The implementation currently supports two data sources:

- 5G-NIDD flow records
- Telecom Italia Milan grid-cell traffic

Both source-specific pipelines produce 64-dimensional embeddings. Milan embeddings have already been passed through the TGNN interface in a real smoke test.

## Completed Coding Work

### 1. 5G-NIDD preprocessing

File: `src/ssl/preprocessing.py`

Implemented:

- Removes the exported `Unnamed: 0` CSV index column.
- Excludes `Label`, `Attack Type`, and `Attack Tool` from SSL pretraining inputs.
- Encodes the 11 categorical context columns.
- Reserves IDs for missing and unseen categorical values.
- Converts masked traffic columns to numeric values.
- Imputes missing numeric values with training-set medians.
- Normalizes numeric features using training-set statistics only.
- Splits NIDD rows by `Seq` order into training and validation partitions.
- Applies deterministic masking using a seed.
- Returns PyTorch tensors for context, masked numeric values, targets, and mask positions.
- Saves and restores fitted preprocessing metadata.

Main output shapes:

```text
context:       [batch_size, 11]
masked values: [batch_size, 14]
targets:       [batch_size, 14]
mask:          [batch_size, 14]
```

### 2. 5G-NIDD encoder and decoder

File: `src/ssl/masked_reconstruction.py`

Implemented:

- `FlowFeatureEncoder` as a real `torch.nn.Module`.
- One categorical embedding table per context column.
- MLP encoder.
- Fixed 64-dimensional embedding output.
- MLP decoder for the 14 masked traffic features.
- Reconstruction loss calculated only over masked positions.
- Input shape and batch-size validation.
- Gradient/backpropagation support.

Model path:

```text
categorical context + masked traffic values
        -> categorical embeddings and MLP encoder
        -> 64-dimensional embedding
        -> MLP decoder
        -> reconstructed traffic values
```

### 3. 5G-NIDD training

File: `src/ssl/train_masked_reconstruction.py`

Implemented:

- Reproducible random seeds.
- Mini-batch `DataLoader` training.
- Adam optimizer.
- Configurable epochs, batch size, mask ratio, hidden size, and device.
- Training and validation reconstruction loss.
- Best-validation-loss checkpoint saving.
- Training metrics saved as JSON.
- Preprocessing metadata saved with the experiment.
- CPU and CUDA device selection.

### 4. 5G-NIDD runtime profiling

File: `src/ssl/profile_masked_reconstruction.py`

Implemented:

- CSV loading timing.
- Preprocessing timing.
- Warm-up inference timing.
- Real encoder/decoder inference timing.
- Reconstruction loss measurement.
- Rows per second.
- Model milliseconds per row.
- End-to-end milliseconds per row.
- JSON profiling report generation.

### 5. 5G-NIDD embedding evaluation

File: `src/ssl/evaluate_embeddings.py`

Implemented:

- Loading a trained SSL checkpoint.
- Generating real 64-dimensional embeddings.
- Training a downstream Logistic Regression classifier.
- Evaluation on checkpoint-held-out sequence rows.
- Micro-F1.
- Macro-F1.
- Accuracy.
- Per-class precision, recall, F1, and support.
- Processed raw-feature baseline.
- Random 64-dimensional embedding baseline.
- Leakage-safe training-to-validation evaluation split.

The old hash-based placeholder evaluator has been removed.

### 6. Milan data adapter

File: `src/ssl/milan_adapter.py`

Implemented for the actual sample files under `data/raw/milan_telecom_italia/`:

- Reads the CSV fields `CellID`, `datetime`, `countrycode`, `smsin`, `smsout`, `callin`, `callout`, and `internet`.
- Aggregates rows from different country codes by `CellID` and `datetime`.
- Produces five canonical traffic fields:
  - `sms_in`
  - `sms_out`
  - `call_in`
  - `call_out`
  - `internet`
- Defines `CellID` as the grid-cell node identifier.
- Preserves `datetime` as the timestamp.
- Splits Milan data chronologically.
- Fits normalization statistics on training intervals only.
- Produces normalized PyTorch feature tensors.

Milan input representation:

```text
CellID + datetime -> five normalized traffic features
```

### 7. Milan SSL encoder and training

File: `src/ssl/milan_ssl.py`

Implemented:

- Five-feature Milan masked autoencoder.
- Configurable masking ratio.
- 64-dimensional Milan embedding output.
- MLP encoder and decoder.
- Chronological training/validation split.
- Adam training.
- Best checkpoint saving.
- Training metrics JSON.
- Preprocessing metadata JSON.
- Embedding export with node and timestamp metadata.

### 8. SSL-to-TGNN bridge

File: `src/ssl/tgnn_bridge.py`

Implemented:

- Validates embedding export metadata.
- Requires exactly 64 embedding dimensions.
- Rejects duplicate `(node_id, timestamp)` records.
- Requires one `source_dataset` per export.
- Checks that every selected node has every selected timestamp.
- Applies deterministic node/timestamp ordering.
- Reshapes embeddings into the TGNN input format:

```text
[N, T, 64]
```

## Important Artifacts

### NIDD smoke-test artifacts

Generated locally from a 2,000-row NIDD training smoke test:

```text
data/processed/ssl_nidd_smoke_test_2000_rows/best_model.pt
data/processed/ssl_nidd_smoke_test_2000_rows/metrics.json
data/processed/ssl_nidd_smoke_test_2000_rows/preprocessing.json
data/processed/ssl_nidd_smoke_test_2000_rows/profiling.json
data/processed/ssl_nidd_smoke_test_2000_rows/evaluation.json
```

These files are local generated artifacts and are not intended for Git commits.

### Milan smoke-test artifacts

Generated from 50,000 aggregated Milan rows:

```text
data/processed/ssl_milan_smoke_test_50000_rows/best_milan_model.pt
data/processed/ssl_milan_smoke_test_50000_rows/metrics.json
data/processed/ssl_milan_smoke_test_50000_rows/preprocessing.json
data/processed/ssl_milan_smoke_test_50000_rows/milan_embeddings.csv
```

The exported CSV contains:

```text
node_id
timestamp
source_dataset
embedding_00 ... embedding_63
```

## Verified Results

### NIDD

The NIDD model was validated with:

```text
embedding output:       [batch_size, 64]
reconstruction output:  [batch_size, 14]
```

A real training smoke test completed successfully, and the checkpoint reload test produced valid 64-dimensional embeddings.

### NIDD profiling

The 2,000-row profiling report recorded approximately:

```text
model time:       0.010026 ms/row
end-to-end time:  0.033089 ms/row
```

These values are CPU smoke-test measurements, not final production benchmarks.

### Milan

A real daily Milan file produced:

```text
aggregated rows: 1,439,976
grid cells:      10,000
time intervals:  144
```

The Milan training smoke test used:

```text
rows:              50,000
training rows:     40,000
validation rows:   10,000
best val. loss:    0.504863
```

A real Milan export was validated with:

```text
embedding window: [29, 5, 64]
TGNN output:      [29, 3]
source_dataset:   milan
```

## How To Verify The Code

### Verify NIDD preprocessing and encoder

```powershell
python -c "import torch; from src.ssl.preprocessing import load_nidd_csv, split_by_sequence, NIDDPreprocessor; from src.ssl.masked_reconstruction import FlowFeatureEncoder; df=load_nidd_csv('data/raw/5g_nidd/Combined.csv', n_rows=1000); train, valid=split_by_sequence(df); prep=NIDDPreprocessor().fit(train); batch=prep.transform_with_mask(valid.head(32), seed=17); output=FlowFeatureEncoder(category_sizes=prep.category_sizes)(batch.context, batch.numeric); assert output.embedding.shape == (32, 64); assert output.reconstruction.shape == (32, 14); print('NIDD Phase 1-2 validation passed')"
```

### Run NIDD training

```powershell
python src/ssl/train_masked_reconstruction.py `
  data/raw/5g_nidd/Combined.csv `
  data/processed/ssl_nidd_training_10000_rows `
  --epochs 5 `
  --batch-size 256 `
  --n-rows 10000
```

### Run NIDD profiling

```powershell
python src/ssl/profile_masked_reconstruction.py `
  data/raw/5g_nidd/Combined.csv `
  data/processed/ssl_nidd_training_10000_rows/best_model.pt `
  data/processed/ssl_nidd_training_10000_rows/profiling.json `
  --n-rows 10000 `
  --batch-size 256
```

### Run NIDD evaluation

```powershell
python src/ssl/evaluate_embeddings.py `
  data/raw/5g_nidd/Combined.csv `
  data/processed/ssl_nidd_training_10000_rows/best_model.pt `
  data/processed/ssl_nidd_training_10000_rows/evaluation.json `
  --n-rows 10000 `
  --batch-size 256
```

### Run Milan training

```powershell
python -c "from src.ssl.milan_ssl import MilanSSLConfig, train_milan; train_milan('data/raw/milan_telecom_italia', 'data/processed/ssl_milan_training', MilanSSLConfig(epochs=5), max_files=7)"
```

### Export Milan embeddings

```powershell
python -c "from src.ssl.milan_ssl import export_milan_embeddings; export_milan_embeddings('data/raw/milan_telecom_italia', 'data/processed/ssl_milan_training/best_milan_model.pt', 'data/processed/ssl_milan_training/milan_embeddings.csv', max_files=7)"
```

### Validate the TGNN bridge

```powershell
python -c "from src.ssl.tgnn_bridge import build_tgnn_window; window=build_tgnn_window('data/processed/ssl_milan_smoke_test_50000_rows/milan_embeddings.csv', node_ids=['1','2','3']); print(window.shape); assert window.shape[2] == 64"
```

## Current Integration Contract

The SSL-to-TGNN handoff is:

```text
node_id:        string
timestamp:      integer epoch milliseconds for Milan
embedding:      float[64]
source_dataset: milan, neversnet5g, or b5g
TGNN tensor:    [N, T, 64]
```

The Milan path has been tested with real embeddings. The existing B5G orchestrator still contains an explicit random SSL placeholder because Milan node IDs do not correspond to B5G graph nodes.

## Remaining Coding Work

These are not part of the completed Review-1 coding handoff, but are the next engineering items:

1. Run larger NIDD training so the evaluation validation split contains enough benign examples.
2. Build a B5G or NeversNet5G source adapter that produces embeddings for the graph nodes used by the orchestrator.
3. Confirm final node ordering and timestep semantics with Thrishala.
4. Replace the B5G orchestrator random SSL embedding function only after compatible graph embeddings are available.
5. Run full-scale experiments and record final metrics separately from smoke-test metrics.
6. Consider a formal NIDD/Milan transfer or joint-training experiment after agreeing on a shared cross-source architecture.

## Git Handoff

The coding work is on:

```text
feature-ssl
```

The completed commits include:

```text
feat(ssl): add NIDD preprocessing pipeline
feat(ssl): implement masked reconstruction encoder
feat(ssl): add masked reconstruction training pipeline
feat(ssl): measure reconstruction inference overhead
feat(ssl): evaluate frozen NIDD embeddings
feat(ssl): add Milan traffic preprocessing adapter
feat(ssl): train and export Milan embeddings
feat(ssl): bridge Milan embeddings into TGNN
```

Generated datasets, checkpoints, profiling reports, and embedding CSVs should remain local and should not be committed unless the team explicitly decides to version a small artifact.
