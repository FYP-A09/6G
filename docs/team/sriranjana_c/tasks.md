# Tasks — Self-Supervised Learning (Sriranjana C)

- [ ] Pull `data/raw/5g_nidd/Combined.csv` (already in repo) and profile the 100+ flow features — decide which are usable per-node signals vs. security-only fields.
- [ ] Get a Milan sample from `data/raw/milan_telecom_italia/` (1-week CSVs) and decide the grid-cell = "node" framing.
- [ ] Prototype the masked-reconstruction pretext task on one dataset; measure reconstruction loss *and* wall-clock training time (this is NFR1 — don't skip the timing).
- [ ] Evaluate embedding quality via the slice-type classification proxy using `deepslice_secure5g/` or `network_slicing_puspakmeher/` labels.
- [ ] Agree the embedding output format/dimension with Thrishala before finalizing the encoder.
- [ ] Prepare the review-1 slide: your 3 papers + identified gap (pretext task design difficulty, compute cost, no DT/TGNN integration yet) + your chosen pretext task and why.
