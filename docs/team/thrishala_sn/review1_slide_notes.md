# Review-1 slide notes — TGNN / Traffic & Topology Prediction (Thrishala S N)

## Your 3 papers + gap (pull from `docs/Literature_Review_Notes.md`)

1. **A Graph Neural Network-Based Digital Twin for Network Slicing Management
   (2022)** — GraphSAGE, <5% E2E latency prediction error, generalizes across
   NSFNET/GEANT2/Synthetic topologies. **Gap**: static graph, ignores temporal
   change, needs labelled data.
2. **Attention to Virtualization: Making Network Digital Twins Aware of Network
   Slicing (2025)** — GraphSAGE + GAT, virtualization-aware, beats RouteNet-Fermi
   on delay prediction. **Gap**: still a static graph, supervised only.
3. **Self-Supervised Spatiotemporal GNNs with Self-Distillation for Traffic
   Prediction (SSGNN, 2023)** — masked reconstruction + GCN + temporal conv +
   self-distillation, up to 5.2% RMSE improvement. **Gap**: road traffic, not 6G
   network traffic; no closed loop into resource allocation.

## Your module's answer to the shared gap

Papers 1-2 are static-graph; paper 3 is dynamic but never closes the loop into a
control decision. This project's TGNN module is explicitly **dynamic** (adjacency
matrix reflects the real-time UE-gNB serving relationship, not a fixed topology)
and its output feeds directly into MARL (§2 of `interface_contracts.md`) — the
prediction isn't the end product, it's the input to a decision.

## What's built and validated (not just designed)

- `src/tgnn/build_graph.py` — validated against real NeversNet5G data: 10 UEs
  correctly matched to their nearest of the 19 real gNodeBs (haversine distance),
  edges weighted by real mean SINR (e.g. `ue_veh10 -> gnb_10`, 11.2 dB) — this is
  the schema Keerthivasan drafted, now run and confirmed to produce a sensible
  graph on real data. Also loads B5G's real GML topologies (416 nodes, 830 edges
  for `graph_0`).
- `src/tgnn/model.py` — a GraphSAGE (mean-aggregation) + temporal-conv model,
  hand-written in plain torch (no torch_geometric installed yet), tested
  end-to-end with tensors shaped exactly like the real interface contract:
  29 nodes × 5 timesteps × 64-d SSL embeddings in, a 3-field prediction
  (`predicted_throughput_bps`/`predicted_latency_ms`/`predicted_load`) out.

## Known open question (see `docs/architecture/graph_schema_draft.md`)

The UE-gNB "serving cell" edge is *reconstructed* via nearest-gNB-by-distance —
the dataset doesn't label it directly. This ignores handover hysteresis and real
SINR-based cell selection; worth comparing against an SINR-based reassignment
rule once there's time, since the two could disagree at cell boundaries.

## Slide checklist

- [ ] 3 papers + gap (above)
- [ ] Show the real sample output from `build_graph.py` (UE → nearest gNB with SINR)
- [ ] Model I/O shapes from `model.py`'s smoke test
- [ ] Flag the reconstructed-edge caveat as an explicit limitation, not hidden
