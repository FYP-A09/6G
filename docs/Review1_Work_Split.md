# Review 1 — Work Split (4 people)

Team: KEERTHIVASAN, KRISH S, SRIRANJANA C, THRISHALA S N.

Split by the four technical pillars of the framework (SSL → TGNN → MARL, all inside
the Digital Twin), so each person's literature review, dataset ownership, and
review-1 deliverable line up with one coherent piece of the system.

| Person | Pillar | Owns from literature review | Owns from datasets | Review-1 deliverable |
|---|---|---|---|---|
| **KRISH S** | MARL / Dynamic Slicing | VNF mobility (DELAY-NSM/R-NSM/FT-NSM), Online Learning for Slicing (OLS/EXP3), Deep RL for E2E Slicing — his existing 4 papers in `Literature_Review_Notes.md` | B5G Network Slicing Dataset (reward/SLA benchmark, over/under-provisioned scenarios) + 6G RAN Telemetry for FL (already has a computed `reward` and `sla_ok_next` column) | Dec-POMDP formulation for the slicing agents, reward function design, PettingZoo + Ray RLlib environment plan |
| **SRIRANJANA C** | SSL (Representation Learning) | Net2Net (position prediction), Encrypted Traffic Classification via SSL, "Revolutionizing Wireless Networks with SSL" survey | 5G-NIDD (real 5G-core flow features), DeepSlice & Secure5G, robertbotez 6G slicing dataset | Pick and justify the pretext task (masked reconstruction vs. contrastive vs. position-prediction) for learning from unlabelled telecom traffic |
| **THRISHALA S N** | TGNN (Traffic / Topology Prediction) | GraphSAGE-based Digital Twin, GAT-based virtualization-aware Digital Twin, Self-Supervised Spatiotemporal GNN (SSGNN) | NeversNet5G (gNB/UE graph, time-varying SINR/CQI/throughput/latency), Telecom Italia Milan (spatial-temporal demand), B5G `graphs/` (topology files) | Graph construction plan (what counts as a node/edge) + PyTorch Geometric Temporal architecture sketch |
| **KEERTHIVASAN** | Digital Twin + Integration | Native Network Digital Twin architecture, AI-Driven DT with NTN integration, Milan-based self-optimizing DT | Repo/pipeline infrastructure, full dataset catalog (`data/raw/README.md`), download scripts (`src/data/download_datasets.py`) | End-to-end architecture diagram tying the three modules together, plus a live repo/dataset readiness walkthrough |

## Shared (not split)

Draft these together since they depend on all four pillars agreeing:

- **Novelty statement**: no surveyed paper combines SSL + TGNN + Digital Twin + MARL for 6G slicing; the recurring gap in existing Digital-Twin/DRL work specifically is that it's *reactive*, not predictive.
- **Work-plan / Gantt slide** for the phase after review 1 (SSL pretext task decision → TGNN integration → MARL environment → closed-loop Digital Twin demo).

## What's already done, per person

Most of the review-1 content already exists in `Literature_Review_Notes.md` — each
person's slide is mainly pulling their 3-4 papers + gap out of that document, stating
which dataset(s) from `data/raw/README.md` they'll use and why, and adding one
sentence on their proposed method for the next phase.
