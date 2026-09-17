# Design — Self-Supervised Learning (Sriranjana C)

## Candidate pretext tasks (from the literature review)
1. **Masked feature/traffic reconstruction** (SSGNN-style) — mask a slice of the traffic
   signal, reconstruct it.
2. **Node Position Prediction / PosPredict** (Net2Net-style) — predict a node's
   position within its own ego-network.
3. **Contrastive learning** across augmented views of the same flow/node.

## Recommended default
**Masked reconstruction**, because it pairs naturally with the TGNN's temporal
convolution stage downstream — SSGNN already fuses both into one architecture, so the
handoff to Thrishala's module is more direct. Treat **position-prediction** as the
strong secondary/ablation to compare against, not a discarded option.

## Encoder
Two variants are needed depending on the data source:
- A **hierarchical GNN encoder** over per-node ego-networks (Net2Net-style) — natural
  fit once a graph exists (e.g. paired with NeversNet5G/B5G topology).
- A **flow-level feature encoder** for 5G-NIDD, which has no inherent graph structure
  until it's paired with topology data.

## Evaluation proxy
Micro-F1 on slice-type classification (using DeepSlice/robertbotez/network_slicing
labels) as a stand-in downstream task — lets you score representation quality before
the full TGNN pipeline exists to test against.

## Output contract
A fixed-size embedding vector per node/timestep, handed to the TGNN as its input
features in place of raw telemetry. Agree the exact dimension with Thrishala before
freezing the encoder architecture.
