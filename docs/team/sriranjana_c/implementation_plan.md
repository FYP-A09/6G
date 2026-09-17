# Implementation Plan - Self-Supervised Learning

Owner: Sriranjana C  
Primary dataset: `data/raw/5g_nidd/Combined.csv`  
Output contract: one 64-dimensional embedding per node/timestep or flow representation, as agreed in `docs/architecture/interface_contracts.md`.

## Current dataset status

The 5G-NIDD file is now available locally at `data/raw/5g_nidd/Combined.csv`.
A sample read shows 52 parsed columns, including the exported `Unnamed: 0` index. The named fields contain the documented traffic, protocol, QoS, and label columns. The context fields have manageable categorical cardinalities, but several fields contain many missing values. The `Label` field is imbalanced, so evaluation splits must be stratified.

## Phase 0 - Contract and environment confirmation

**Goal:** Freeze the input/output assumptions before training.

Tasks:

- Confirm with Thrishala that the embedding dimension remains `D=64`.
- Confirm the tensor shape and node/timestep ordering expected by the TGNN.
- Decide whether 5G-NIDD embeddings are used for flow-level pretraining only, or are aggregated into node/timestep embeddings for downstream TGNN input.
- Record the preprocessing seed, feature list, and model configuration.

Deliverable: a short agreed SSL-to-TGNN contract note.

Difficulty: Low.  
Dependency: Thrishala's confirmation is needed for the final handoff, but implementation can begin with the existing `D=64` contract.

## Phase 1 - Data preprocessing

**Goal:** Convert raw NIDD rows into reproducible model inputs.

Tasks:

- Drop `Unnamed: 0`; retain `Seq` only as an ordering key, not as a learned feature unless explicitly justified.
- Exclude `Label`, `Attack Type`, and `Attack Tool` from pretraining inputs.
- Encode the 11 context columns with per-column vocabularies and an unknown/missing category.
- Convert masked traffic columns to numeric values and fill or flag missing values consistently.
- Fit normalization statistics on the training partition only.
- Implement deterministic masking over the volume and rate/load columns.
- Return context tensors, numeric tensors, target tensors, and a boolean mask.
- Split by sequence/time ordering where possible to avoid leakage between train and validation rows.

Deliverable: reusable dataset/preprocessing class with a small-batch smoke test.

Difficulty: Medium.  
Can start: Immediately.

## Phase 2 - Encoder and decoder

**Goal:** Replace the placeholder with a real PyTorch model.

Tasks:

- Make `FlowFeatureEncoder` an `nn.Module`.
- Add one embedding table per categorical context column.
- Concatenate categorical embeddings with masked numeric features.
- Add an MLP encoder ending in a 64-dimensional embedding.
- Add an MLP decoder reconstructing the masked traffic features.
- Compute reconstruction loss only at masked positions.
- Expose a forward path returning both the embedding and reconstruction.
- Add shape, missing-value, and output-dimension tests.

Deliverable: model that accepts a batch and returns `[batch_size, 64]` embeddings plus reconstructions.

Difficulty: High.  
Can start: Immediately after the preprocessing interface is defined.

## Phase 3 - Masked-reconstruction training

**Goal:** Train meaningful embeddings rather than only exercising the forward pass.

Tasks:

- Start with a small reproducible subset of NIDD.
- Train with Adam and the configured mask ratio.
- Track training and validation reconstruction loss.
- Save the best checkpoint and preprocessing vocabularies/statistics together.
- Repeat with at least one alternate mask ratio or hidden dimension as a small ablation.
- Record CPU/GPU, batch size, number of rows, elapsed time, and peak memory if available.

Deliverable: trained checkpoint, configuration file, and loss/runtime table.

Difficulty: High.  
Can start: After Phases 1 and 2.  
Dependency: No teammate blocker; the local `torch`, `pandas`, and `sklearn` packages are available.

## Phase 4 - Compute-overhead measurement

**Goal:** Satisfy NFR1 with real model measurements.

Tasks:

- Update `evaluate_reconstruction_loss` to run preprocessing, masking, encoder, decoder, and loss calculation.
- Report total elapsed time, rows/second, milliseconds/row, and reconstruction loss.
- Separate first-run/warm-up time from steady-state timing where practical.
- Compare the real forward-pass timing against the current I/O-only baseline.

Deliverable: reproducible profiling output and a short paragraph for the review report.

Difficulty: Medium.  
Dependency: Phase 3 model implementation.

## Phase 5 - Representation evaluation

**Goal:** Measure whether the learned embedding is useful downstream.

Tasks:

- Replace the hash-based placeholder in `evaluate_embeddings.py`.
- Do not pass unrelated labelled-dataset schemas directly into the NIDD encoder.
- Define a compatible feature adapter or use an explicitly compatible labelled evaluation dataset.
- Train a simple classifier on frozen embeddings.
- Use stratified train/test splitting because the observed NIDD label distribution is imbalanced.
- Report Micro-F1 and at least one imbalance-aware metric such as Macro-F1 or per-class F1.
- Compare against a raw-feature baseline and a dummy/random embedding baseline.

Deliverable: evaluation table with dataset, representation, Micro-F1, Macro-F1, and split details.

Difficulty: High.  
Dependency: Requires a clear evaluation-schema decision; the current placeholder pipeline is not a valid final experiment.

## Phase 6 - Milan generalization

**Goal:** Demonstrate SSL beyond one traffic source.

Tasks:

- Obtain the one-week Milan sample from Thrishala or Keerthivasan.
- Aggregate the five traffic fields by grid cell and 10-minute interval.
- Treat each grid cell as a node and preserve the timestamp.
- Define an adapter from Milan's five numeric features into the shared embedding interface.
- Evaluate transfer or joint pretraining between NIDD and Milan.
- Document which features are shared, source-specific, and normalized separately.

Deliverable: Milan preprocessing adapter and a two-source generalization result.

Difficulty: Medium to High.  
Blocked until: the Milan sample is available and the grid-cell representation is agreed with Thrishala.

## Phase 7 - TGNN handoff and integration

**Goal:** Replace random SSL embeddings in the end-to-end pipeline.

Tasks:

- Export embeddings with `node_id`, `timestamp`, `embedding`, and `source_dataset`.
- Map the exported embeddings into the TGNN ordering `[N, T, 64]`.
- Replace `make_placeholder_ssl_embeddings` in `src/digital_twin/orchestrator.py`.
- Run the TGNN smoke test and the B5G orchestration path.
- Verify that no random placeholder embeddings remain in the real path.
- Version the checkpoint and preprocessing artifacts together with the interface schema.

Deliverable: real SSL-to-TGNN integration run and updated integration notes.

Difficulty: Medium.  
Dependency: trained checkpoint, stable node/timestep mapping, and Thrishala's confirmation of the input contract.

## Phase 8 - Review-1 evidence package

**Goal:** Prepare the material that can be shown and defended.

Include:

- Three-paper comparison and research gap.
- Why masked reconstruction was selected over contrastive learning and position prediction.
- NIDD feature and masking diagram.
- Model architecture diagram.
- Reconstruction-loss plot.
- Runtime/overhead measurement.
- Embedding evaluation table.
- Limitations: flow-level NIDD has no native topology, Milan is a separate source, and the first embedding dimension is an agreed starting point rather than an optimized result.

Difficulty: Medium.  
Can proceed in parallel with implementation using the existing draft slide notes.

## Dependency summary

Can begin immediately:

- Phases 0 through 4, using NIDD and small local samples.
- Review-1 literature and architecture slides.
- Synthetic-batch tests for the encoder.

Requires another person or shared data:

- Final tensor/node ordering: coordinate with Thrishala.
- Milan preprocessing and two-source generalization: wait for the Milan sample.
- Full end-to-end TGNN integration: wait for a trained checkpoint and stable handoff mapping.

The main immediate milestone is a trained NIDD masked-reconstruction encoder with measured runtime. Milan and TGNN integration should be treated as the next milestone, not prerequisites for starting the SSL implementation.
