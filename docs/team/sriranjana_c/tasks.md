# Tasks — Self-Supervised Learning (Sriranjana C)

## Done (drafted/prototyped ahead of your validation pass — react to these, don't treat them as final)

- [x] 5G-NIDD feature profile (50 columns, 1.2M rows, grouped into context vs.
      masking-target sets) — `dataset_notes.md`.
- [x] Masked-reconstruction encoder and decoder — `src/ssl/masked_reconstruction.py`,
      with categorical context embeddings, masked-only reconstruction loss, and
      the agreed D=64 output.
- [x] NIDD embedding evaluation — `src/ssl/evaluate_embeddings.py`, with
      leakage-free classifier scoring and raw/random baselines.
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
- [x] Fill in the actual `nn.Module` encoder in `masked_reconstruction.py`,
      including categorical embeddings for the `CONTEXT_COLUMNS` and the fixed
      64-dimensional output.
- [x] Run the real reconstruction profiling once the encoder exists, and
      report the wall-clock number (NFR1 — don't skip the timing). The Phase 4
      replacement is now implemented in `src/ssl/profile_masked_reconstruction.py`;
      the smoke-test report is `data/processed/ssl_nidd_smoke_test_2000_rows/profiling.json`.
- [x] Replace the placeholder evaluator with checkpoint-backed NIDD embedding
      evaluation in `src/ssl/evaluate_embeddings.py`, including Micro-F1,
      Macro-F1, per-class F1, raw-feature, and random-embedding baselines.
      **Now run at full scale**: trained the encoder on the complete real
      1,215,890-row `Combined.csv` (10 epochs, `reports/nidd_ssl_full/`,
      validation loss ~0.0005–0.001), replacing the old 2,000-row smoke
      checkpoint. Found and fixed a real evaluation bug along the way: the
      classifier's train/test split reused the SSL training's *sequential*
      split (`split_by_sequence`), but `Combined.csv` is concatenated in large
      contiguous same-attack-type blocks — a sequential split put 99.88% of one
      class in the test tail (first checked the truncated file: 50 Benign vs.
      39,950 Malicious; then the full file: 242,884 Benign vs. 294 Malicious,
      the *opposite* skew), collapsing accuracy to ~1%, *worse* than the
      majority-class baseline. Fixed `evaluate_embeddings.py` to use a
      stratified random split by label for the classifier evaluation instead
      (training itself still correctly uses the sequential split). Real,
      properly-balanced result (`reports/nidd_ssl_full/evaluation_stratified.json`,
      243,178 real held-out flows, 95,547 Benign / 147,631 Malicious):
      **SSL embeddings — accuracy 0.7664, Macro-F1 0.7083, Micro-F1 0.7664**
      (Benign precision 0.996/recall 0.407, Malicious precision 0.722/recall
      0.999) essentially matches processed raw features (accuracy 0.7656,
      Macro-F1 0.7068) while compressing 50 raw columns into a fixed 64-D
      embedding, and clearly beats the random-embedding baseline (accuracy
      0.6071, Macro-F1 0.3778, which fails on Benign entirely). Honest
      interpretation: the encoder preserves essentially all the raw features'
      discriminative signal — it hasn't yet demonstrated an accuracy edge
      *over* raw features, and Benign recall (0.407) is the weak point worth
      improving (mask-ratio/loss-weighting tuning, not yet attempted).
- [ ] Run a true NIDD+Milan transfer or joint-pretraining experiment. The two
      sources currently use source-specific adapters because NIDD is flow-level
      with categorical protocol fields while Milan is numeric grid-cell traffic;
      do not combine their raw tensors without an agreed shared adapter.
- [x] Agree the embedding output format/dimension with Thrishala — D=64, fixed
      and consistent across `masked_reconstruction.py`, `milan_ssl.py`,
      `neversnet_ssl.py`, `tgnn_bridge.py`, `model.py`'s `SSL_EMBEDDING_DIM`, and
      `docs/architecture/interface_contracts.md` §1. Confirmed no mismatch across
      any of these files as of this review; revisit only if a future ablation
      argues for a different D.
- [x] Add the SSL-to-TGNN bridge in `src/ssl/tgnn_bridge.py`. It validates the
      64-dimensional export and reshapes Milan embeddings into `[N, T, 64]`; a
      real 29-node x 5-timestep Milan window was consumed by `TGNNPredictor`.
      The B5G orchestrator remains on its explicit placeholder until a
      B5G/NeversNet5G-compatible embedding export and final node ordering are
      agreed with Thrishala.
- [x] Add the NeversNet5G telemetry adapter, masked-reconstruction encoder,
      sequential training, logging, checkpointing, and 64-D export in
      `src/ssl/neversnet_ssl.py`. It processes one UE CSV at a time and was
      validated on a real sample file; full-dataset execution requires the
      teammate's complete part folders.
