# Tasks — TGNN / Traffic & Topology Prediction (Thrishala S N)

- [ ] Extract the NeversNet5G graph structure (gNB locations + UE trajectories) — full
      data lives at `E:\FYP DATA\6G\data\raw\neversnet5g\data\` (too large for the git
      repo; re-download with `python src/data/download_datasets.py neversnet5g_full`
      if working from a fresh machine). Define the node/edge schema first.
- [ ] Extract the B5G topology files at
      `data/raw/b5g_slicing/2.2.1-10kprocessed/graphs/*.txt` (GML format) as a second
      topology source for the generalization test (FR3).
- [ ] Prototype a GraphSAGE + temporal-conv baseline using PyTorch Geometric Temporal
      on one topology before scaling to both.
- [ ] Agree the SSL embedding input format with Sriranjana; agree the prediction
      output format with Krish.
- [ ] Prepare the review-1 slide: your 3 papers + identified gap (static graphs,
      over-smoothing, label dependency) + why GraphSAGE + Temporal-Conv addresses it.
