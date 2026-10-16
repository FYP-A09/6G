# Unified data, synthetic dataset, and why MARL

Response to the supervisor's review comments (12 Oct 2026):

1. Look for unified data that connects everything.
2. Why MARL, why not a single agent?
3. If unified data is not available, create it synthetically.
4. Which datasets do the papers use?

Numbers in this document come from `reports/unified/` and can be regenerated with
the commands at the end.

## 4. Which datasets do the papers use?

All 16 papers in `docs/Literature_Review_Notes.md`:

| # | Paper | Data used | Kind |
|---|---|---|---|
| 1 | Ali & Arslan (2025), AI-driven DTs for slicing with NTNs | Custom Python 5G simulator, 50 UEs, 20 MHz | Own simulator |
| 2 | Li et al. (2026), multi-agent learning for RAN slicing | NS-3 RAN slicing simulator | Own simulator |
| 3 | Chowdhury et al. (2025), semantic slicing framework | None (conceptual, Python/TensorFlow) | None |
| 4 | John (2026), GAN-enhanced DRL for semantic slicing | Custom 6G slicing simulator, GAN-generated traffic | Own simulator |
| 5 | MARL for vehicular slicing (2023) | Mininet emulator + Ryu SDN controller | Own emulator |
| 6 | Native network DT architecture for 6G (2025) | Operator-inspired signalling-storm scenario, no public data | Private |
| 7 | Attention to virtualization, slicing-aware DTs (2025) | Docker UPF measurements + extended OMNeT++ | Own testbed + simulator |
| 8 | GNN-based DT for slicing management (2022) | NSFNET (14 nodes), GEANT2 (24), synthetic 50-node topologies | Public topologies + synthetic |
| 9 | SSL for wireless networks, survey (2024) | None (survey) | None |
| 10 | Net2Net, SSL network representation (2025) | Cora, Citeseer, DBLP, PubMed, GitHub, Facebook, LastFM Asia, Twitch | Public graph benchmarks (not telecom) |
| 11 | Encrypted traffic classification with SSL (2022) | Self-generated SDN traffic (Mininet + D-ITG) | Own generated |
| 12 | Network slice mobility for 6G (2022) | Discrete-time simulation | Own simulator |
| 13 | Slicing for AI (2025) | Mobile-health dataset, 245,921 samples | Public, not telecom |
| 14 | Self-supervised spatiotemporal GNN for traffic (2023) | METR-LA, PEMS-BAY, PEMS03/04/07/08 | Public road traffic |
| 15 | DRL for end-to-end slicing (2023) | OpenAirInterface / FlexRAN testbed | Own testbed |
| 16 | Self-optimizing 6G via network DT intelligence (2026) | Telecom Italia Milan | Public cellular traffic |

Summary: 8 of 16 generate their data with their own simulator, emulator or
testbed, one more builds synthetic topologies, three use non-telecom benchmarks,
one uses private operator data, and two use none. Only one uses public cellular
traffic (Milan, which we also use). **No paper uses a public dataset that unifies
topology, per-slice traffic, QoS and allocation decisions**; generating the data
is the normal practice in this literature.

## 1. Is there unified data?

What "unified" needs, one record joining: network topology, per-cell traffic over
time, per-slice demand, QoS (latency, jitter, loss), SLA outcome, and the
allocation action with its reward.

| Dataset | Topology | Traffic over time | Slice types | QoS | Actions / reward |
|---|---|---|---|---|---|
| Telecom Italia Milan (we hold, 62 days) | Grid layout only | Yes, 10-min, 10,000 cells | Services (internet/call/SMS), not slices | No | No |
| NeversNet5G (we hold) | gNB/UE positions | Yes, per UE | No | SINR/CQI/throughput | No |
| B5G slicing, Farreras et al. 2024 (we hold) | Transport topologies | No (static samples) | eMBB/URLLC/mMTC | Recorded `delta` only; latency/jitter/loss null in 0 of 4,350 sampled slices | Admission scenarios, no action log |
| 5G-NIDD (we hold) | No | Flows | No | No | No |
| 6G RAN telemetry FL, Kaggle (we hold) | No | Yes, 200 clients x 150 windows | Slice mix proportions | PRB load, p50/p95 latency, SLA flag | Reward column, no actions |
| DeepSlice / Kaggle slicing sets (we hold) | No | Day/time fields | Slice label | Packet loss and delay budgets | No |
| Botez et al. 6G slicing (we hold) | No | No | 6G slice classes | Latency/jitter/loss budgets | No |
| O-RAN slicing traces (e.g. Colosseum/ORANSlice work) | Single-cell testbeds | Yes | eMBB/URLLC | PRB, throughput, latency | Release not confirmed |

Web search (Oct 2026) found nothing that combines all six. The closest public
sources are the Farreras B5G set, which is transport-only with no RL action
logs, and O-RAN slicing traces, which are single-cell with no confirmed data
release. So the answer to (1) is no, and we followed (3).

## 3. The synthetic unified dataset

We built one, grounded in real data wherever real data exists (`src/unified/`):

| Piece | Source |
|---|---|
| Traffic per cell, every 10 minutes | Real: Telecom Italia Milan |
| Topology | Real: the Milan grid's cell layout and geographic adjacency |
| Slice demand | Milan's three services mapped to slices: internet → eMBB, calls → URLLC, SMS → mMTC |
| Latency, jitter vs load | Calibrated: curves fitted to the public 6G RAN telemetry dataset (30,000 windows) |
| Loss, SLA | Rules from the same SLA targets as our B5G environment |
| Interference between cells | Modelled: capacity falls with neighbours' load |
| Actions, rewards | From an environment that can be run live or logged to disk |

Two products:

- **Short dataset for the experiments** (`src/unified/dataset.py`): 7 days x 100 cells,
  100,700 rows, with the TGNN forecast and a logged behaviour policy. Sample:
  `reports/unified/unified_dataset_sample.csv`.
- **Digital Twin dataset** (`src/unified/twin_dataset.py`), 393.6 million rows x 122 columns in two builds:
  - **Core year:** 364 days x 400 cells x 6 scenarios (baseline, events, 6G growth, energy saving, industrial URLLC, massive IoT).
  - **Full city:** all 10,000 Milan cells x 62 recorded days x 3 scenarios.

  It also carries injected events (flash crowds, outages, interference, URLLC bursts, mMTC storms,
  degradations), multi-horizon forecasting targets, 3GPP CQI/MCS, EARTH energy, and noisy
  telemetry with gaps. Datasheet: `docs/Unified_Twin_Dataset.md`.

**Is it realistic?** Two checks (`reports/unified/dataset_checks.json`):

- Traffic keeps Milan's real daily cycle (about 19 Mbps per cell at 5 am, about 51 Mbps
  in the afternoon), because it is Milan's traffic.
- eMBB SLA rate against load, compared with the reference dataset's own SLA label.
  With a deterministic latency model the match was poor (mean gap 0.086: SLA flipped
  from always met to never met at 68% load). Adding per-window latency variation
  (one parameter, σ = 0.21, fitted to that reference curve) brings the mean gap to
  **0.021** across 15 load bins, e.g. 0.934 vs 0.919 at 57% load and 0.231 vs 0.264 at
  77%. Because σ was fitted on this curve, this is a goodness-of-fit, not an independent
  test.

Limits: QoS is a calibrated model rather than a packet-level simulation; the
reference telemetry is itself simulated by its authors; the slice mapping and
interference coupling are assumptions, kept in one config (`ModelConfig`) so they
can be varied.

## 2. Why MARL and not a single agent?

We tested it on the unified environment. Both methods are DDPG with the same reward,
the same training budget (300 episodes of 6 hours, 10,300 updates) and the same test
days (the last 2 of 7 real Milan days, never seen in training). 3 seeds each, at 9, 25,
49 and 100 cells.

- **Single agent:** one centralised actor sees every cell (state = N x 17) and outputs
  every cell's slice shares (N x 4). Its learning rate was tuned first: 5 settings on a
  validation day, best actor lr = 1e-4 (`reports/unified/single_agent_sweep.json`).
- **MARL:** one actor shared by all cells, each acting on its own 17 local features;
  a critic that also sees the mean of its grid neighbours' observations and actions;
  a team reward that mixes each cell's reward with its neighbours'.

Held-out reward (mean ± sd over 3 seeds; 1.0 = every SLA met and all demand served):

| Cells | MARL | Single agent (tuned) | Equal split | Expert rules, current demand | Expert rules + TGNN forecast | Expert rules + perfect forecast |
|---|---|---|---|---|---|---|
| 9 | **0.767 ± 0.018** | 0.411 ± 0.038 | 0.761 | 0.783 | 0.842 | 0.929 |
| 25 | **0.765 ± 0.042** | 0.300 ± 0.027 | 0.731 | 0.755 | 0.802 | 0.898 |
| 49 | **0.758 ± 0.049** | 0.275 ± 0.014 | 0.712 | 0.734 | 0.782 | 0.876 |
| 100 | **0.769 ± 0.030** | 0.273 ± 0.002 | 0.713 | 0.739 | 0.790 | 0.880 |

| Cells | MARL actor parameters | Single-agent actor parameters | Single-agent input / output size |
|---|---|---|---|
| 9 | 19,332 | 114,468 | 153 / 36 |
| 25 | 19,332 | 200,548 | 425 / 100 |
| 49 | 19,332 | 329,668 | 833 / 196 |
| 100 | 19,332 | 604,048 | 1,700 / 400 |

Per-slice SLA rate at 100 cells: MARL eMBB 0.57, URLLC 0.98, mMTC 0.39, demand served
0.89; single agent eMBB 0.21, URLLC 0.35, mMTC 0.24, demand served 0.27.

What this shows:

1. **The single agent does not scale.** Its input and output grow with the number of
   cells (400 outputs at 100 cells) and so does its network (114k → 604k parameters).
   Its reward falls from 0.41 to 0.27 as the network grows and it is below the
   equal-split baseline at every size. It mostly learns to leave capacity idle.
2. **MARL does not depend on network size.** The shared actor has the same 19,332
   parameters at 9 or 100 cells, every cell's experience trains it (more cells = more
   data, not a harder problem), and its reward stays at 0.76–0.77 at every size.
   A trained actor can be applied to a network of any size without retraining.
3. **It fits how a RAN is run.** Each cell decides from its own measurements and its
   neighbours', so there is no central controller to collect the whole network's state
   every 10 minutes and no single point of failure.
4. **MARL learns the expert rules, but not more.** It matches or slightly beats the
   hand-written expert policy on current demand (within one standard deviation), and it
   stays below the expert policy given the TGNN forecast (0.79–0.84).

Honest limits:
- Equal training budget. A centralised agent trained much longer, or with a
  factorised action space, may do better; with the same budget it does not.
- Decision time is not a differentiator at this size: both take about 0.1–0.2 ms
  per decision on a CPU.
- Forecast as an MARL input (25 cells, 3 seeds): no forecast 0.778 ± 0.039, TGNN
  forecast 0.765 ± 0.042, perfect forecast 0.762 ± 0.047. The forecast does not help
  the learned agent at this training budget, while the same forecast lifts the expert
  rules by about 0.05 at every size. So today the TGNN's value is shown through the
  expert controller; making MARL use it is the next step.

TGNN forecast quality on the test days (mean squared error, Mbps², next 10 minutes):
eMBB 53.9 vs 61.2 for persistence (12% better), URLLC 0.92 vs 1.30 (29%), mMTC 0.52 vs
0.71 (27%).

**Does the forecaster generalise to the whole city?** At first, no. The forecaster
trained on the 100 central cells was applied to all 10,000 Milan cells on the same two
held-out days. On the 9,600 quiet outer cells its eMBB error was 10.8x that of persistence,
because it had learned the centre's traffic level and over-predicted those cells by about 6 Mbps.

We fixed the cause rather than the symptom (`src/unified/forecast_city.py`):
- each cell's traffic is divided by its own training-period 95th percentile
- every training step uses a random 32 x 32 patch of the grid

Same model, same training days. Result, `reports/unified/forecast_city.json`, improvement over
persistence on the held-out days:

| Cells | Block forecaster (eMBB / URLLC / mMTC) | City forecaster (eMBB / URLLC / mMTC) |
|---|---|---|
| 100 block cells | +12% / +30% / +27% | **+24% / +36% / +27%** |
| 300 cells around the block | -8% / -30% / -34% | **+19% / +30% / +19%** |
| 9,600 outer cells | -985% / -101% / -100% | **+19% / +27% / +11%** |
| All 10,000 cells | -635% / -74% / -75% | **+19% / +29% / +14%** |

The city forecaster is also better on the block it was not specifically trained for, and
its bias on the outer cells drops from +6.2 to +0.16 Mbps. The Slice Twin app's City
view uses it.

Figures: `docs/figures/unified_marl_vs_single.png`, `unified_prediction_value.png`,
`unified_forecast_ablation.png`.
