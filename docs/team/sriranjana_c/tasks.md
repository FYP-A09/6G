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

- [ ] Get a Milan sample from `data/raw/milan_telecom_italia/` (1-week CSVs) and decide the grid-cell = "node" framing (not yet covered by the drafted code above).
- [ ] Fill in the actual `nn.Module` encoder in `masked_reconstruction.py` (the
      `FlowFeatureEncoder.encode()` method is currently a `NotImplementedError`
      placeholder) — decide categorical-embedding sizes for the `CONTEXT_COLUMNS`.
- [ ] Run `evaluate_reconstruction_loss` for real once the encoder exists, and
      report the wall-clock number (NFR1 — don't skip the timing).
- [ ] Re-run `evaluate_embeddings.py` with the real encoder in place of the
      placeholder and report the actual Micro-F1 (currently only exercises the
      scoring pipeline, not real representation quality).
- [ ] Agree the embedding output format/dimension with Thrishala — already fixed
      at D=64 in both `masked_reconstruction.py` and `docs/architecture/interface_contracts.md`; flag if that turns out to be too small/large once real training starts.
