# Tasks — TGNN / Traffic & Topology Prediction (Thrishala S N)

**Data split:** you prototype on *samples* — Keerthivasan holds the full NeversNet5G
(28GB) and full Milan (20GB) on his machine (`E:\FYP DATA\6G\`) and owns scaling your
design up to the full data once it's validated. You get:
- `Thrishala_S_N_neversnet5g_sample.zip` — one part-folder (`part1` + `part1_5`,
  ~6.4GB raw), enough UE trajectories + gNB signal data to build and test the graph
  construction logic.
- `Thrishala_S_N_partial.zip` — the Milan 1-week sample (7 daily CSVs, 643MB).

## Done (drafted/prototyped ahead of your validation pass — react to these, don't treat them as final)

- [x] Graph structure extraction — `src/tgnn/build_graph.py`, tested against the
      real NeversNet5G part1 sample (10 UEs correctly matched to 19 real gNodeBs by
      nearest-distance, weighted by real SINR) and against B5G's real GML topology
      (416 nodes). **Validate the nearest-gNB heuristic** — it's a reconstruction
      (the dataset doesn't label serving cell directly), flagged as an open question
      in `graph_schema_draft.md` and `review1_slide_notes.md`.
- [x] GraphSAGE + temporal-conv baseline — `src/tgnn/model.py`, plain torch (no
      torch_geometric installed yet), runs end-to-end against dummy tensors shaped
      like the real contract (29 nodes × 5 timesteps × 64-d embeddings in, 3-field
      prediction out). Needs real training data plumbed through once Sriranjana's
      encoder produces real embeddings instead of dummy ones.
- [x] Review-1 slide notes — `review1_slide_notes.md`.

## Still yours to do

- [ ] Decide whether the nearest-gNB heuristic in `build_graph.py` is good enough,
      or needs replacing with an SINR-based reassignment rule — compare the two on
      the sample before Keerthivasan scales either approach to the full 28GB.
- [x] Extract B5G topology files as a *second* topology source for the
      generalization test (FR3) — confirmed directly: the same `TGNNPredictor`
      (no code or shape changes) processes 4 different real B5G graphs
      (graph_0: 416 nodes/830 edges, graph_1: 238/480, graph_5: 361/730,
      graph_10: 51/100) and produces a correctly-shaped `[N, 3]` prediction for
      each. Generalizes across topology size without retraining or reshaping.
- [ ] Swap PyTorch Geometric Temporal in for the hand-rolled `GraphSAGELayer` in
      `model.py` once that dependency is installed — same math, less code, worth
      doing before scaling past the smoke test.
- [ ] Once Sriranjana's SSL encoder produces real embeddings (not dummy tensors),
      plug them into `model.py` and train on an actual next-step prediction target.
