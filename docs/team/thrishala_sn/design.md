# Design — TGNN / Traffic & Topology Prediction (Thrishala S N)

## Architecture
GraphSAGE-style spatial aggregation (neighbor sampling, mean aggregation — proven in
the GraphSAGE-DT paper) combined with a temporal gated convolution or GRU over the
time axis (the SSGNN pattern). This gives a **GraphSAGE + Temporal-Conv hybrid**,
with the GAT attention variant available as a swap-in if over-smoothing shows up in
practice (per the virtualization-aware DT paper).

## Input
Sriranjana's SSL-pretrained node embeddings, plus raw structural features (edge type
— gNB-gNB vs. gNB-UE — and distance/mobility state).

## Graph definition
- **Nodes**: `{gNB, UE}` for NeversNet5G, `{network element}` for B5G.
- **Edges**: radio link or routing link.
- **Edge weights**: from SINR/CQI (NeversNet5G) or routing capacity (B5G).

## Training objective
Multi-step-ahead regression on next-window per-node/per-slice traffic and latency,
using Log-Cosh loss (per the GraphSAGE-DT paper).

## Regularization
Self-distillation (per SSGNN) as an optional addition once the base model is stable —
targets FR3 (generalizing across topologies) specifically.

## Output contract
A per-node (or per-slice-aggregated) predicted-demand vector, timestamped, handed to
the MARL module as part of agent state. Agree the exact format with Krish before
freezing this.
