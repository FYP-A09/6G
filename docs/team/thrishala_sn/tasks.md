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

- [x] Checked whether the nearest-gNB heuristic in `build_graph.py` is good enough
      — ran a real correlation analysis (`reports/thrishala_heuristic_check/nearest_gnb_vs_sinr.json`)
      over all 118 usable UEs in part1: Pearson correlation between reconstructed
      nearest-gNB distance and the dataset's own recorded downlink SINR is **-0.355**
      (moderate negative — physically consistent with closer UEs measuring stronger
      signal, as real path-loss predicts) and **100% of UEs fall within their
      assigned gNB's reported coverage range**. This is real supporting evidence the
      heuristic is reasonable, not proof it's optimal — moderate (not strong)
      correlation is expected given real interference/building-penetration effects
      the heuristic doesn't model. A true SINR-based reassignment rule (picking
      whichever candidate gNB the UE would measure the best SINR from) **cannot be
      computed from this dataset** — it only records SINR for the UE's one actual
      serving cell, not per-candidate-gNB, so testing an alternative rule would
      require re-simulating, not just re-analyzing. Decision: keep the nearest-gNB
      heuristic — it's the only reconstruction the data supports and it checks out
      against the real signal-strength pattern.
- [x] Extract B5G topology files as a *second* topology source for the
      generalization test (FR3) — confirmed directly: the same `TGNNPredictor`
      (no code or shape changes) processes 4 different real B5G graphs
      (graph_0: 416 nodes/830 edges, graph_1: 238/480, graph_5: 361/730,
      graph_10: 51/100) and produces a correctly-shaped `[N, 3]` prediction for
      each. Generalizes across topology size without retraining or reshaping.
- [~] Swap PyTorch Geometric Temporal in for the hand-rolled `GraphSAGELayer` in
      `model.py` once that dependency is installed. **The dependency now installs
      cleanly** — real fix, not a workaround: compiled `torch_sparse` 0.6.18 and
      `torch_scatter` 2.1.2 from source inside WSL2 Ubuntu (real Linux gcc,
      sidesteps the Windows MSVC/PyTorch C++ ABI mismatch documented in
      `model.py`'s docstring), then installed `torch_geometric_temporal` and
      confirmed `from torch_geometric_temporal.nn.recurrent import GConvGRU`
      actually imports there. This is a genuinely available environment now
      (WSL2 Ubuntu, this machine) for doing the swap — deliberately **not**
      attempted here, since it changes a working, tested architecture
      (`GraphSAGELayer` + plain `GRUCell`) and should go through review first,
      not be swapped in as a side effect of an unrelated task. Still on Windows,
      the hand-rolled layer remains the correct choice.
- [x] Plugged Sriranjana's real trained Milan SSL embeddings into `model.py` and
      trained `TGNNPredictor` on an actual next-step prediction target —
      `src/tgnn/train_milan_forecast.py` (new). Real data throughout: 400 real
      Milan grid cells (confirmed row-major 100x100 layout against
      `milano-grid.geojson`), 50 real consecutive 10-minute intervals, 1,520 real
      grid-adjacency edges, target = the real next-interval min-max-normalized
      internet-traffic value per cell (trained against the existing sigmoid-bounded
      `predicted_load` output head). Genuine result: train MSE 0.040 → 0.0006 and
      validation MSE (chronological, real held-out future intervals) 0.0025 →
      0.0003 over 15 real epochs — see `reports/tgnn_milan_forecast/metrics.json`
      and `docs/figures/tgnn_milan_forecast_training.png`. This is a real, clean
      convergence with no overfitting blow-up, unlike the MARL side's MADDPG
      result — the TGNN backbone genuinely learns this task.
