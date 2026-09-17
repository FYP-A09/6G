# Tasks — Self-Supervised Learning (Sriranjana C)

## Done (drafted/prototyped ahead of your validation pass — react to these, don't treat them as final)

- [x] 5G-NIDD feature profile (50 columns, 1.2M rows, grouped into context vs.
      masking-target sets) — `dataset_notes.md`.
- [x] Masked-reconstruction pretext task skeleton — `src/ssl/masked_reconstruction.py`,
      flow-level encoder variant (5G-NIDD has no inherent graph), fixed at the
      agreed D=64 output. The profiling hook (`evaluate_reconstruction_loss`) runs
      today and measures I/O timing; the actual encoder forward pass is a `TODO`
      pending a decision on categorical-column embedding sizes.
- [x] Slice-type classification evaluation proxy — `src/ssl/evaluate_embeddings.py`,
      runs end-to-end today (sklearn Logistic Regression + Micro-F1) against a
      placeholder embedding function — swap in the real encoder once trained.
- [x] Review-1 slide notes — `review1_slide_notes.md`.

## Still yours to do

- [x] Get the Milan sample from `data/raw/milan_telecom_italia/` and define
      each `CellID` as a grid-cell node with `datetime` as the timestamp. The
      source-specific adapter is implemented in `src/ssl/milan_adapter.py` and
      aggregates country rows into five normalized traffic features.
- [x] Implement the Milan masked-reconstruction encoder and trainer in
      `src/ssl/milan_ssl.py`, including chronological validation, checkpoint
      saving, and 64-dimensional embedding export. The bounded smoke-test
      artifacts are under `data/processed/ssl_milan_smoke_test_50000_rows/`.
- [ ] Fill in the actual `nn.Module` encoder in `masked_reconstruction.py` (the
      `FlowFeatureEncoder.encode()` method is currently a `NotImplementedError`
      placeholder) — decide categorical-embedding sizes for the `CONTEXT_COLUMNS`.
- [x] Run the real reconstruction profiling once the encoder exists, and
      report the wall-clock number (NFR1 — don't skip the timing). The Phase 4
      replacement is now implemented in `src/ssl/profile_masked_reconstruction.py`;
      the smoke-test report is `data/processed/ssl_nidd_smoke_test_2000_rows/profiling.json`.
- [x] Replace the placeholder evaluator with checkpoint-backed NIDD embedding
      evaluation in `src/ssl/evaluate_embeddings.py`, including Micro-F1,
      Macro-F1, per-class F1, raw-feature, and random-embedding baselines. The
      current 2,000-row smoke checkpoint evaluates its held-out sequence split;
      final metrics require a larger checkpoint with enough benign validation
      samples.
- [ ] Run a true NIDD+Milan transfer or joint-pretraining experiment. The two
      sources currently use source-specific adapters because NIDD is flow-level
      with categorical protocol fields while Milan is numeric grid-cell traffic;
      do not combine their raw tensors without an agreed shared adapter.
- [ ] Agree the embedding output format/dimension with Thrishala — already fixed
      at D=64 in both `masked_reconstruction.py` and `docs/architecture/interface_contracts.md`; flag if that turns out to be too small/large once real training starts.
