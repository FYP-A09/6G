# Review-1 slide notes — Self-Supervised Learning (Sriranjana C)

## Your 3 papers + gap (pull from `docs/Literature_Review_Notes.md`)

1. **Net2Net: Self-Supervised Network Representation Learning via Position
   Prediction (2025)** — hierarchical GNN encoder + PosPredict pretext task.
   **Gap**: static graph representations, no temporal network evolution.
2. **Encrypted Network Traffic Classification in SDN using SSL (2022)** —
   two-stage SSL (pretrain on unlabelled, fine-tune on small labelled set), 97.64%
   accuracy, +2% over supervised. **Gap**: task-specific to traffic
   classification, not designed for graph-based network optimization.
3. **Revolutionizing Wireless Networks with SSL (2024, survey)** — Generative/
   Contrastive/Predictive SSL paradigms. **Gap**: designing efficient pretext
   tasks for wireless data remains hard; SSL+DT+TGNN+RL integration largely
   unexplored.

## Your module's answer to the shared gap

All three either work on static graphs, are task-specific, or stop at "SSL is
promising" without integrating it into a larger pipeline. This project's SSL
module produces embeddings specifically shaped to feed the TGNN (fixed D=64
vector, see `docs/architecture/interface_contracts.md` §1) — SSL isn't the end
goal here, it's the first stage of a predictive pipeline.

## Chosen pretext task and why

**Masked reconstruction** (mask the volume/rate-load feature groups per flow,
reconstruct from protocol/context columns) — chosen over pure contrastive
learning and over Net2Net's position-prediction because it pairs naturally with
the TGNN's temporal-convolution stage downstream (same pattern SSGNN uses).
Position-prediction remains a documented secondary approach for ablation, not
discarded (see `design.md`).

## What's built so far

- `docs/team/sriranjana_c/dataset_notes.md` — full 5G-NIDD column profile (50
  columns, 1.2M rows), with columns grouped into context vs. masking-target sets.
- `src/ssl/masked_reconstruction.py` — the pretext-task skeleton (flow-level
  encoder variant, since 5G-NIDD has no inherent graph), including a runnable
  profiling hook for NFR1 (measuring compute overhead — the SSL survey paper's
  own flagged gap).
- `src/ssl/evaluate_embeddings.py` — the Micro-F1 evaluation proxy against the
  slice-labeled Kaggle datasets already in the repo, runnable end-to-end today
  with a placeholder embedding function (swap in the real encoder once trained).

## Slide checklist

- [ ] 3 papers + gap (above)
- [ ] Pretext task choice + why (above)
- [ ] Column profile summary from `dataset_notes.md`
- [ ] NFR1 compute-overhead measurement, once the real encoder is trained
