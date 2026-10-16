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
src/ssl/                SSL encoders (5G-NIDD, Milan, NeversNet5G) + embedding evaluation
src/tgnn/               Graph construction, TGNNPredictor, Milan forecast training
src/marl/               B5G slicing environment, MADDPG, evaluation, tuning
src/digital_twin/       Orchestrator closed loop, telemetry replay, Simu5G scenario config, result figures
reports/                Real outputs of the runs described below (metrics, evaluations, small checkpoints)
docs/architecture/      Module interface contracts, system diagram, graph schema draft
docs/team/<person>/     Per-person requirements / design / tasks (the tasks.md files hold the detailed evidence)
notebooks/              Exploration notebooks
docs/                   Literature review notes and meeting minutes
```

## Current status

Everything below was run on real data; numbers come from `reports/` and the
`docs/team/*/tasks.md` files. Open items are listed honestly at the end.

| Module | What exists | Result |
|---|---|---|
| SSL (5G-NIDD) | Masked-reconstruction encoder, 64-D embeddings, trained on all 1.2M flows | Stratified held-out split: accuracy 0.766, Macro-F1 0.708 — matches raw features (0.766 / 0.707), beats random embeddings (0.607 / 0.378); Benign recall is weak (0.41) |
| SSL (Milan) | Masked-reconstruction encoder on the full-scale release | Trains; embeddings feed the TGNN |
| TGNN | GraphSAGE + GRU `TGNNPredictor`; next-step forecast on 400 real Milan cells over a full day | Beats "predict the last value" by about 14% (MSE 0.000616 vs 0.000720, 28 validation windows, one seed). Feeding raw features instead of SSL embeddings scores the same (0.000612), so SSL adds nothing here — see `docs/team/thrishala_sn/tasks.md` |
| MARL | B5G environment, MADDPG, train/validate/test tuning search | Trained actor 0.447 vs heuristic 0.372 vs equal-split 0.312 on a fresh test block; the reward is nearly trivial (B5G has no latency/loss data), so treat as a working pipeline, not a strong policy |
| Digital Twin | Orchestrator: B5G sample → TGNN → MARL action → twin approve/reject gate | Runs end to end; approves 0 actions because B5G lacks the telemetry the SLA check needs |
| Unified environment (`src/unified/`) | Real Milan traffic + real grid adjacency, QoS calibrated on the 6G RAN telemetry dataset, load-coupled interference; one record per cell per 10 min | Answers the 12 Oct review: no public dataset unifies traffic, topology, slices, actions and SLA, so we built one — see [`docs/Unified_Data_and_MARL.md`](docs/Unified_Data_and_MARL.md) |
| MARL vs single agent | Shared-actor MADDPG vs tuned centralised DDPG, 9-100 cells, 3 seeds, held-out days | MARL 0.76-0.77 at every size with a constant 19k-parameter actor; single agent falls from 0.41 to 0.27 (below equal split). MARL matches the expert rules but stays below expert rules + TGNN forecast (0.79-0.84) |
| TGNN forecast value | Next-10-min per-slice forecast on the 100-cell grid | Beats persistence by 12% / 29% / 27% (eMBB / URLLC / mMTC); lifts the expert controller by about 0.05 reward at every size; does **not** help MARL as an extra input (0.765 vs 0.778 without) |
| City-wide forecaster | `src/unified/forecast_city.py`: same TGNN, per-cell scaling, trained on random 32x32 patches of the grid | Beats persistence on all 10,000 cells by 19% / 29% / 14%; the block-trained model did not transfer (10.8x persistence's error on quiet cells) |
| Digital Twin dataset | `src/unified/twin_dataset.py`: core year (400 cells x 364 days x 6 scenarios) + full city (10,000 cells x 62 days x 3 scenarios) | 393.6 million rows x 122 columns on the project drive; datasheet [`docs/Unified_Twin_Dataset.md`](docs/Unified_Twin_Dataset.md) |
| Simulator | OMNeT++ 6.4.0 + INET 4.7.0 + Simu5G 1.7.0 built in WSL2; real NR VoIP run | 28,092 events, MOS 4.41/5, 0 loss — the project's own 19-gNodeB scenario is **not** wired in yet |

### Reproducing

Run the SSL scripts directly rather than importing them (`src/ssl` shadows
Python's standard-library `ssl` module):

```bash
python src/ssl/train_masked_reconstruction.py <5g_nidd Combined.csv> reports/nidd_ssl_full --epochs 10 --batch-size 512
python src/ssl/evaluate_embeddings.py <Combined.csv> reports/nidd_ssl_full/best_model.pt reports/nidd_ssl_full/evaluation_stratified.json
python src/ssl/milan_ssl.py <milan full-scale data dir> reports/milan_ssl_full --max-files 1 --max-rows 500000 --epochs 8
python src/tgnn/train_milan_forecast.py <milan data dir> reports/milan_ssl_full/best_milan_model.pt reports/tgnn_milan_forecast --max-files 1 --max-rows 500000
python src/marl/tune_maddpg.py          # needs the full B5G release; edit ROOT at the top
python src/unified/experiments.py   # unified env: heuristics, TGNN, MARL vs single agent -> reports/unified/results.json
python src/unified/figures.py
python src/unified/forecast_city.py    # city-wide TGNN forecaster -> reports/unified/forecast_city.json
python src/unified/twin_dataset.py --block 20 --days 364    # Digital Twin dataset (needs the Milan files)
python -m pytest tests/ -q
```

### Known issues and open work

- B5G slice numbers repeat across slice types; the environment keys agents by
  slice-list position (a bug here once invalidated every MARL number — fixed, with a regression test).
- SLA thresholds are not calibrated (needs the Farreras et al. definition of `delta`); the reward's
  latency / packet-loss / SLA terms are inactive because B5G never populates those fields.
- SSL is trained per dataset; no joint NIDD + Milan representation yet.
- NeversNet5G UE identities cannot be linked across the four main parts (dataset limitation).
- `torch_geometric_temporal`'s `GConvGRU` does not install on Windows; it does build in WSL2 Ubuntu, but
  `TGNNPredictor` still uses a plain `GRUCell`.
- The orchestrator still uses placeholder SSL embeddings for B5G and an untrained TGNN.

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
