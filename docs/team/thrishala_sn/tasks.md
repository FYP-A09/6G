# Tasks — TGNN / Traffic & Topology Prediction (Thrishala S N)

**Data split:** you prototype on *samples* — Keerthivasan holds the full NeversNet5G
(28GB) and full Milan (20GB) on his machine (`E:\FYP DATA\6G\`) and owns scaling your
design up to the full data once it's validated. You get:
- `Thrishala_S_N_neversnet5g_sample.zip` — one part-folder (`part1` + `part1_5`,
  ~6.4GB raw), enough UE trajectories + gNB signal data to build and test the graph
  construction logic.
- `Thrishala_S_N_partial.zip` — the Milan 1-week sample (7 daily CSVs, 643MB).

## Tasks
- [ ] Unzip the NeversNet5G sample and extract the graph structure (gNB locations +
      UE trajectories) from `part1`. Define the node/edge schema — this schema is
      what Keerthivasan will apply to the full 8-part dataset later, so make it
      general, not part1-specific.
- [ ] Extract the B5G topology files at
      `data/raw/b5g_slicing/2.2.1-10kprocessed/graphs/*.txt` (GML format, already in
      the repo via Krish's zip) as a second topology source for the generalization
      test (FR3).
- [ ] Prototype a GraphSAGE + temporal-conv baseline using PyTorch Geometric Temporal
      on the sample data before Keerthivasan scales it to the full 28GB/20GB sets.
- [ ] Agree the SSL embedding input format with Sriranjana; agree the prediction
      output format with Krish.
- [ ] Prepare the review-1 slide: your 3 papers + identified gap (static graphs,
      over-smoothing, label dependency) + why GraphSAGE + Temporal-Conv addresses it.
