# End-to-end system diagram

For the review-1 slide. Three-layer structure per the Native NDT architecture
paper, with this project's three intelligence modules filling the NDT Layer.

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. NETWORK ELEMENT LAYER (physical / simulated)                         │
│                                                                           │
│   19 gNodeBs (X2 mesh, Nevers, France) ── real UE trajectories (SUMO)   │
│   src/digital_twin/simu5g_scenario/  ←→  data/raw/neversnet5g/          │
│                                                                           │
└───────────────────────────────┬───────────────────────────────────────┬─┘
                                 │ telemetry (SINR, CQI, throughput,      │ approved
                                 │ latency, position — real-time sync)    │ actions
                                 ▼                                       │
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. NDT LAYER — this project's contribution                              │
│                                                                           │
│   ┌───────────────┐      ┌────────────────┐      ┌──────────────────┐  │
│   │ SSL            │      │ TGNN            │      │ MARL              │  │
│   │ (Sriranjana C) │ ───▶ │ (Thrishala S N) │ ───▶ │ (Krish S)         │  │
│   │                │emb-  │                 │demand│                   │  │
│   │ masked recon-  │edding│ GraphSAGE +     │pred- │ Dec-POMDP,        │  │
│   │ struction on   │ (§1) │ temporal-conv,  │iction│ MADDPG / SF-DTMC, │  │
│   │ unlabelled     │      │ dynamic graph   │ (§2) │ reward penalizes  │  │
│   │ traffic        │      │ over gNB/UE     │      │ SLA violation +   │  │
│   │                │      │                 │      │ reconfig churn    │  │
│   └───────────────┘      └────────────────┘      └────────┬─────────┘  │
│                                                             │candidate   │
│                                                             │action (§3) │
│                                                             ▼            │
│                                              ┌──────────────────────┐   │
│                                              │ Twin evaluation       │   │
│                                              │ (what-if, before      │   │
│                                              │ physical rollout)     │   │
│                                              │ — Keerthivasan         │   │
│                                              └──────────┬────────────┘   │
└─────────────────────────────────────────────────────────┼──────────────┘
                                                            │ approved only
                                                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 3. APPLICATION LAYER                                                     │
│    PRB allocation · VNF scaling/migration · slice-level SLA enforcement │
└─────────────────────────────────────────────────────────────────────────┘
```

`(§1)`/`(§2)`/`(§3)` refer to the matching sections in
[`interface_contracts.md`](interface_contracts.md).

## The one-sentence novelty claim this diagram is drawing

Every reviewed Digital Twin / DRL paper closes the loop **without** the SSL→TGNN
predictive stage — MARL (or plain DRL) acts on the current observation only. This
diagram's left-to-right flow through SSL and TGNN *before* MARL sees anything is
the entire novelty argument in one picture.
