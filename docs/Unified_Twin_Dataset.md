# Unified Digital Twin dataset

One dataset that joins everything the separate public datasets could not:
- real traffic on the real Milan cell layout
- per-slice demand and multi-horizon forecasting targets
- allocation actions, and the QoS, SLA, radio and energy outcomes that follow
- injected incidents
- what-if scenarios

There is one row per cell per 10 minutes.

Generator: `src/unified/twin_dataset.py`. Outcomes come from `src/unified/env.py`.
It exists because no public dataset unifies these (see `docs/Unified_Data_and_MARL.md`),
and 8 of the 16 papers we reviewed also generate their own data.

## Two builds

| Build | Cells | Days | Scenarios | Rows | Columns |
|---|---|---|---|---|---|
| **Core year** | 400 (20 x 20 block, central Milan) | 364 (1 Nov 2013 - 30 Oct 2014) | 6 | 125,796,000 | 122 |
| **Full city** | 10,000 (the whole Milan grid) | 62 (every recorded day) | 3: baseline, events, energy_saving | 267,810,000 | 122 |
| **Total** | | | | **393,606,000** | |

Full data (too large for git) is on the project drive, as Parquet with zstd compression:
- Core year: `E:\FYP DATA\6G\data\processed\unified_twin\timeseries\<scenario>\week_NN.parquet` (52 files per scenario).
- Full city: `E:\FYP DATA\6G\data\processed\unified_twin\city_62d\timeseries\<scenario>\day_NNN.parquet` (62 files per scenario, about 1.44 million rows each).

Read a file with `pandas.read_parquet(path)`. For many files, use `pyarrow.dataset.dataset(folder)` to filter
on columns or rows without loading everything. Measured on a test build, the data takes about 260
bytes per row.

Committed in `reports/unified_twin/` (core) and `reports/unified_twin/city_62d/` (city):
- `cells.csv`: topology, position, zone, industrial flag, neighbours, base SINR
- `calendar.csv`: every day, its source day and growth factor
- `events.csv`: every injected event, with cells and parameters
- `summary.json`: row counts, SLA rates, reward, energy and per-event statistics for each scenario

## What is real, calibrated and modelled

| Part | Status | Source / rule |
|---|---|---|
| Traffic per cell per 10 min, the 62 recorded days | **Real** | Telecom Italia Milan, all 10,000 cells, local time (CET) |
| Cell positions, adjacency, distance to the Duomo | **Real** | `milano-grid.geojson`, 4-connected grid |
| Holidays (incl. Sant'Ambrogio, Milan's 7 Dec holiday) | **Real calendar** | Italian public holidays |
| Days 63-364 (core year) | Bootstrapped from real days | Same-weekday real day (holidays reuse real holidays; the Christmas week is not used for ordinary days) x (1 + 0.20 x day/365) growth x per-cell day factor exp(N(0, 0.05)) x interval noise exp(N(0, 0.03)) |
| Slice demand | Modelled mapping | internet → eMBB, calls → URLLC, SMS → mMTC. One Mbps scale for both builds, fitted on the central 20 x 20 block (each slice's 95th percentile = 80% / 10% / 5% of a 100 Mbps cell). Outer cells are therefore lightly loaded, as suburban macro cells are |
| Latency p50/p95, jitter vs utilisation | **Calibrated** | Curves fitted to the 6G RAN telemetry dataset (30,000 windows), with per-window lognormal variation (σ = 0.21) fitted to that dataset's SLA-vs-load curve: `src/unified/qos_calibration.json` |
| Inter-cell interference | Modelled | capacity = 100 Mbps / (1 + 0.3 x mean neighbour load) |
| Loss, SLA | Modelled rules | overflow + residual loss; targets eMBB 50 ms / 20 ms / 0.99, URLLC 5 ms / 1 ms / 0.99999, mMTC 100 ms / 50 ms / 0.99 |
| Spectral efficiency, modulation | Standard table | 3GPP TS 38.214 Table 5.2.2.1-2 (CQI 1-6 QPSK, 7-9 16QAM, 10-15 64QAM) |
| Power and energy | Standard model | EARTH macro base station (Auer et al., 2011): 6 x (130 W + 4.7 x 20 W x load); 6 x 75 W asleep; 0 W in outage |
| Users, packets, queue, SINR, handovers | Derived | formulas below |
| Industrial zone | **Assumption** | 12% of cells, three times likelier in the outer ring (only used by `industrial_urllc`) |
| Events | Injected | schedule below; every event is listed in `events.csv` |

## Scenarios

All scenarios share the same days, the same behaviour policies and the same noise. A difference
between two scenarios is caused by the scenario alone. Every scenario except `baseline` also
carries the injected events.

| Scenario | What changes | Core year: eMBB / URLLC / mMTC SLA | Reward | Energy |
|---|---|---|---|---|
| `baseline` | recorded traffic, no events | 0.704 / 0.731 / 0.651 | 0.763 | 3,473 MWh |
| `events` | + flash crowds, outages, interference, URLLC bursts, mMTC storms, degradations | 0.703 / 0.730 / 0.651 | 0.763 | 3,473 MWh |
| `growth_6g` | eMBB x1.3, URLLC x2, mMTC x3 | 0.553 / 0.696 / 0.610 | 0.688 | 3,648 MWh |
| `energy_saving` | 01:00-05:00: lightly loaded cells (< 15 Mbps) on one colour of a checkerboard sleep and hand all their traffic to awake neighbours | 0.703 / 0.737 / 0.659 | 0.766 | 3,435 MWh (-1.1%) |
| `industrial_urllc` | industrial cells: URLLC x4 on weekday shifts (06:00-22:00), x1.5 otherwise | 0.693 / 0.723 / 0.646 | 0.756 | 3,481 MWh |
| `massive_iot` | mMTC x10, and x3 again in the first 10 minutes of every hour (synchronised reporting) | 0.554 / 0.710 / 0.524 | 0.660 | 3,628 MWh |

Full city (62 days):

| Scenario | eMBB / URLLC / mMTC SLA | Reward | Energy | Cells asleep |
|---|---|---|---|---|
| `baseline` | 0.970 / 0.750 / 0.681 | 0.861 | 12,164 MWh | 0 |
| `events` | 0.970 / 0.750 / 0.681 | 0.860 | 12,164 MWh | 0 |
| `energy_saving` | 0.970 / 0.765 / 0.700 | 0.868 | 11,775 MWh (-3.2%) | 7,080,848 cell-intervals (7.9%) |

How to read these numbers:
- Events cover only 0.16% of rows, so the scenario totals barely move. Their effect is in the per-event table below.
- eMBB does better city-wide (0.97) than in the core (0.70), because most of the city is quiet.
- Energy saving saves 3x more energy across the city than in the dense core, because quiet suburbs can sleep.
- A sleeping cell carries no traffic, so it trivially meets its SLA. Compare slice SLA on awake cells (`cell_asleep == False`) when judging the cost of sleeping.

## Events

Rates are per day per 400 cells; the city build scales them by the number of cells (x25).

| Type | Rate / day | Duration | Cells | Effect |
|---|---|---|---|---|
| flash_crowd | 0.5 | 1-3 h, starts 17:00-22:00 | grid radius 1-2 | eMBB x2-4, URLLC x1.5-2, mMTC x1.2 |
| cell_outage | 0.2 | 30 min-4 h | 1 | capacity 0; 70% of its traffic moves to neighbours (`rehomed_*`) |
| interference_spike | 0.3 | 1-6 h | radius 1 | interference coupling x3 |
| urllc_burst | 0.3 | 20-60 min | 1-3 | URLLC x3-6 |
| mmtc_storm | 0.3 | 10-30 min | radius 2 | mMTC x5-10 (synchronised IoT reporting, a signalling storm) |
| capacity_degradation | 0.1 | 2-12 h | 1 | capacity x0.5 |

Injected: 569 events in the core year, 2,667 in the city build.

What each event does (rows inside the event, `events` scenario):

| Event | Core: eMBB / URLLC / mMTC SLA | Core reward | City: eMBB / URLLC / mMTC SLA | City reward |
|---|---|---|---|---|
| none | 0.704 / 0.730 / 0.651 | 0.764 | 0.970 / 0.750 / 0.681 | 0.861 |
| flash_crowd | 0.217 / 0.683 / 0.598 | 0.532 | 0.883 / 0.747 / 0.670 | 0.830 |
| cell_outage | 0.000 / 0.095 / 0.019 | 0.019 | 0.000 / 0.080 / 0.034 | 0.013 |
| interference_spike | 0.577 / 0.697 / 0.629 | 0.704 | 0.959 / 0.748 / 0.682 | 0.856 |
| urllc_burst | 0.608 / 0.510 / 0.575 | 0.631 | 0.967 / 0.651 / 0.688 | 0.819 |
| mmtc_storm | 0.658 / 0.730 / 0.344 | 0.679 | 0.967 / 0.748 / 0.499 | 0.814 |
| capacity_degradation | 0.323 / 0.710 / 0.647 | 0.578 | 0.900 / 0.733 / 0.660 | 0.829 |

Each event hits the slices it should:
- flash crowds and degradations hurt eMBB
- URLLC bursts hurt URLLC
- mMTC storms hurt mMTC
- outages take everything down

In the city, the same events do less harm, because they mostly land on cells with spare capacity.

## Actions (behaviour policies)

Each cell follows one behaviour policy per day:

| Policy | Probability | What it does |
|---|---|---|
| `headroom` | 0.45 | Expert rule: URLLC and mMTC sized to their SLA-safe utilisation, the rest to eMBB |
| `headroom_explore` | 0.25 | 70% expert + 30% random |
| `proportional` | 0.12 | Shares in proportion to demand |
| `equal_split` | 0.08 | A third to each slice |
| `random` | 0.10 | Dirichlet draw, can leave PRBs idle |

The mix gives offline-RL methods varied actions to learn from.

## Columns (122)

**Identity and calendar (20):**
`scenario, interval, day_index, date, interval_of_day, hour, minute, weekday, month,
day_of_year, iso_week, season, is_weekend, is_holiday, is_business_hours, is_night,
source (milan_real | bootstrap), source_date, cell_id, zone (centre < 2 km | inner < 5 km | outer)`.

**Control (3):**
`policy, idle_share, reconfiguration_churn`. The slice shares and PRBs are per-slice columns, listed below.

**Events and state (8):**
- `event_type_code` (0 none, 1 flash_crowd, 2 cell_outage, 3 interference_spike, 4 urllc_burst, 5 mmtc_storm, 6 capacity_degradation, 7 cell_sleep)
- `event_type`, `event_id` (joins `events.csv`), `anomaly`, `cell_asleep`
- `capacity_factor`, `interference_factor`, `anomaly_score` (EWMA z-score of total demand)

**Cell radio and capacity (11):**
`capacity_mbps, spare_capacity_mbps, total_demand_mbps, total_served_mbps, prbs_used, load,
sinr_db, cqi, modulation, spectral_efficiency_bps_hz, all_slices_sla_ok`.

**Neighbourhood (6):**
`neighbour_load, neighbour_mean_demand_mbps, neighbour_all_sla_ok, rehomed_in_mbps,
rehomed_out_mbps, handover_users_est`.

**Energy (3):**
`power_w, energy_wh, energy_efficiency_mbit_per_j`.

**Per slice, 21 fields x 3 slices = 63** (prefix `eMBB_`, `URLLC_`, `mMTC_`):
- control: `share`, `prbs` (share x 106 PRBs)
- demand and forecasting targets: `demand_mbps`, `demand_next_10min_mbps`, `demand_next_30min_mbps`, `demand_next_60min_mbps`, `demand_1h_mean_mbps`, `demand_delta_mbps`
- users and capacity: `users_est`, `allocated_mbps`, `utilisation`, `served_mbps`, `throughput_per_user_mbps`
- traffic: `packets`, `queue_kbit`
- QoS and SLA: `latency_p50_ms`, `latency_p95_ms`, `jitter_ms`, `loss`, `sla_margin_ms`, `sla_ok`

**Reward (2):**
`reward, team_reward`.

**Telemetry as a monitor sees it (6):**
- `obs_total_demand_mbps`, `obs_embb_latency_p95_ms`, `obs_load`: 5%, 5% and 3% noise
- `obs_sinr_db` (±1 dB), `obs_cqi`
- `telemetry_missing`: 1% of rows, 50% during an outage or sleep

Derived-field formulas:
- **Users:** demand / (4, 0.2, 0.02) Mbps per user (eMBB, URLLC, mMTC).
- **Packets in 10 minutes:** served Mbps x 600 s / packet size. Packet sizes: eMBB 1,500 B (MTU), URLLC 32 B (3GPP TR 38.913), mMTC 125 B.
- **Queue:** by Little's law, served rate x p50 delay (Mbps x ms = kbit).
- **SINR:** base SINR per cell ~ N(14.4, 5.1) dB, the reference telemetry's mean and spread. Effective SINR comes from the capacity loss via Shannon: η = (capacity / 100) x log2(1 + SINR0).
- **CQI:** standard SINR thresholds (-6.7 ... 22.7 dB) map SINR to CQI 0-15. Spectral efficiency and modulation then follow from the 38.214 table.
- **Handovers:** users moved by an outage or sleep, = re-homed eMBB traffic / 4 Mbps.
- **Energy:** power x 10 min. Efficiency = served Mbit / J.
- **Interval 0** of each scenario is not written: a row needs the previous decision.

## Checks run on the generated data

On a 3-day test build of all six scenarios:
- Every scenario has identical (interval, cell) rows.
- Energy saving conserves traffic exactly: 14,521 Mbps handed in = 14,521 handed out. Sleeping cells carry zero demand and draw 450 W, only between 01:00 and 05:00.
- Industrial URLLC is exactly x1.5 off-shift; massive-IoT mMTC is exactly x10, and x30 on the hour.
- `telemetry_missing` = 1.0% of rows.
- `demand_next_10min` and `demand_next_60min` equal the cell's demand 1 and 6 rows later.

The realism check on the calibrated QoS is in `docs/Unified_Data_and_MARL.md` (SLA-vs-load gap 0.021).

## Known limits

- The bootstrapped part of the year reuses November–December patterns. It has no real summer behaviour (for example, Milan's August drop).
- QoS is a calibrated model, not a packet-level simulation. The reference telemetry used for calibration is itself simulated by its authors.
- The slice mapping, the industrial zone, the per-user rates and the packet sizes are assumptions. They are kept in one place in the generator so they can be varied.

## Uses

- **SSL:** pretraining on the `obs_*` telemetry with missing values; anomaly detection against `anomaly`.
- **TGNN:** forecasting `demand_next_{10,30,60}min` on the real grid graph (`cells.csv`). The city build tests spatial generalisation (see the city forecaster in `docs/Unified_Data_and_MARL.md`).
- **MARL / offline RL:** states, logged actions from five behaviour policies, rewards, `team_reward`.
- **Digital Twin:** what-if comparisons of the six scenarios on the same days and cells, e.g. the energy saved by night sleep against the SLA cost on neighbours.

## Regenerate

You need the Milan files on E:. On the first run, the generator also builds a 1 GB disk cache of the whole grid.

```bash
python src/unified/twin_dataset.py --block 20 --days 364                                        # core year, 6 scenarios, about 30 min
python src/unified/twin_dataset.py --block 100 --days 62 --scenarios baseline,events,energy_saving  # full city, about 40 min
python src/unified/twin_dataset.py --block 100 --days 62 --scenarios growth_6g,massive_iot          # more city scenarios (about 22 GB each)
```
