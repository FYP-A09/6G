# Requirements — TGNN / Traffic & Topology Prediction (Thrishala S N)

## Functional
- **FR1** Predict future per-node/per-slice traffic demand and/or QoS (latency,
  throughput) over a defined horizon, consuming Sriranjana's SSL-pretrained
  embeddings as node features.
- **FR2** Handle a **dynamic** (time-varying) graph, not a static one — this is the
  #1 gap flagged against both the GraphSAGE-DT and GAT-DT papers, which used static
  graphs.
- **FR3** Generalize to unseen topologies (per the GraphSAGE-DT paper's transfer
  results across NSFNET/GEANT2/Synthetic) — evaluate on at least two different
  topology sources.
- **FR4** Prediction output is directly consumable as MARL agent state — coordinate
  the exact format/units with Krish.

## Non-functional
- **NFR1** Avoid the over-smoothing / poor long-term dependency behavior flagged
  against conventional GNNs (the motivation behind the GAT-DT paper) — justify
  whatever architecture choice is made against this specific failure mode.
- **NFR2** Prediction latency must fit inside the MARL control loop's budget (see
  Krish's NFR1) — TGNN inference sits on the critical path, not just an offline
  analysis step.

## Data requirements
- An actual graph topology with time-varying node signals → **NeversNet5G**
  (`data/raw/neversnet5g/`, real gNB/UE graph with SINR/CQI/throughput/latency time
  series) and **B5G** `graphs/` + `slices/` (`data/raw/b5g_slicing/`, topology +
  per-flow QoS).
- A large spatial-temporal demand signal for pretraining/validation →
  **Telecom Italia Milan** (`data/raw/milan_telecom_italia/`).
