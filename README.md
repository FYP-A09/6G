# Digital Twin Assisted Dynamic Network Slicing for AI-Native 6G Networks

Final-year project: a Digital Twin sandbox that combines **Self-Supervised
Learning (SSL)**, a **Temporal Graph Neural Network (TGNN)**, and **Multi-Agent
Reinforcement Learning (MARL)** to move eMBB/URLLC/mMTC network slicing from
reactive to predictive.

Pipeline: raw telemetry → SSL pretraining (learn representations from
unlabelled traffic) → TGNN (predict future per-node/per-slice traffic &
topology state) → MARL agents (allocate resources per slice against the
predicted state) → Digital Twin (safe sandbox to test allocation policies
before touching the physical network).

## Repo layout

```
data/raw/               Source datasets (small ones committed; large ones gitignored — see data/raw/README.md)
data/processed/         Cleaned/aligned data derived from raw (gitignored)
src/data/               Dataset download & preprocessing scripts
src/digital_twin/       Network Element Layer simulator config (Simu5G scenario)
docs/architecture/      Module interface contracts, system diagram, graph schema draft
docs/team/<person>/     Per-person requirements / design / tasks
notebooks/              Exploration notebooks
docs/                   Literature review notes and meeting minutes
```

## Getting started

```bash
python src/data/download_datasets.py all   # robertbotez + b5g + the 5 Kaggle datasets
```

See [`data/raw/README.md`](data/raw/README.md) for what each dataset contains,
what's deferred due to disk space (NeversNet5G full, Milan full), and which
pipeline stage (SSL / TGNN / MARL) each one is best suited for.

## Project docs

- [`docs/Literature_Review_Notes.md`](docs/Literature_Review_Notes.md) — 16 papers reviewed across Digital Twin, SSL, GNN/TGNN, and MARL/DRL for slicing.
- [`docs/Progress_Notes.md`](docs/Progress_Notes.md) — running log of meetings and progress against the 11 Jul 2026 kickoff action items.
- [`docs/Review1_Work_Split.md`](docs/Review1_Work_Split.md) — the 4-person work split for review 1, with per-person `docs/team/<person>/{requirements,design,tasks}.md`.
- [`docs/architecture/`](docs/architecture/) — module interface contracts, the end-to-end system diagram, and a draft graph schema.

## Novelty position

No surveyed paper combines all four pillars (SSL + TGNN + Digital Twin + MARL)
for 6G slicing; the recurring gap across Digital-Twin/DRL papers is that they
react to observed state rather than predicting it. This project's claimed
contribution is inserting an SSL→TGNN predictive stage ahead of the MARL
controller, validated inside a Digital Twin sandbox before any policy touches
a live network.
