# Review-1 slide notes — Digital Twin + Integration (Keerthivasan)

## Your 3 papers + gap (pull from `docs/Literature_Review_Notes.md`)

1. **Native Network Digital Twin Architecture for 6G: From Design to Practice
   (2025)** — 3-layer architecture (Network Element / NDT / Application),
   gray-box modeling. **Gap**: monitoring/verification only, no GNN/TGNN/SSL/RL,
   no autonomous optimization.
2. **AI-Driven Digital Twins: Optimizing 5G/6G Network Slicing with NTNs
   (Ali & Arslan, 2025)** — real-time DT sync feeding a DDPG agent. **Gap**:
   assumes clean telemetry, purely reactive, no predictive topology mapping.
3. **Toward Self-Optimizing 6G Networks Through Network Digital Twin
   Intelligence (2026)** — AI-native DT on the real Telecom Italia Milan dataset,
   rolling-average traffic prediction + Isolation Forest + MLP. **Gap**: classical
   ML instead of graph learning, no RL-based closed loop.

**Shared thread across all three**: every one of them either monitors/simulates
without closing the loop, or closes the loop reactively (current observation only,
no prediction). Neither failure mode is addressed by adding more compute to the
same architecture — it needs the SSL→TGNN stage ahead of the controller, which is
exactly what this project adds.

## How this project's Digital Twin differs

- Full 3-layer NDT lifecycle (Preparation → Creation → Runtime → Feedback), same
  as the Native NDT paper, but with the NDT Layer actually populated by three
  learned modules (SSL, TGNN, MARL) instead of left as a monitoring shell.
- "What-if" evaluation gate between MARL's candidate action and physical rollout
  (see `docs/architecture/interface_contracts.md`, §3) — no policy reaches the
  physical network without twin approval first.
- Grounded in a real Simu5G scenario (19-gNodeB mesh, Nevers, France —
  `src/digital_twin/simu5g_scenario/`) rather than a toy simulator, addressing the
  "avoid oversimplified network models" instruction from the 11 Jul MoM.

## Slide checklist

- [ ] 3 papers + shared gap (above)
- [ ] End-to-end architecture diagram (`docs/architecture/system_diagram.md`)
- [ ] One line per module: SSL (Sriranjana), TGNN (Thrishala), MARL (Krish) — what
      each does and which dataset backs it
- [ ] Repo/dataset readiness: 5 datasets committed to `github.com/FYP-A09/6G`, 3
      more distributed as per-person zips, full NeversNet5G + Milan on hand for
      scaling once designs are validated on samples
- [ ] Novelty statement (shared across all 4 — see `docs/Review1_Work_Split.md`)
