# NeversNet5G Data Description

## 1. Overview

NeversNet5G contains processed event-level per-UE metric traces generated from a
SUMO and Simu5G/OMNeT++ co-simulation over Nevers, France. The release focuses on
the per-UE CSV dataset described in the dataset paper.

This release does not include the downstream edge-level filled graph used by the
companion routing study. That representation can be derived from the per-UE CSVs
by applying task-specific aggregation, alignment, and filling procedures.

## 2. Scenario Parts

| Part | Files | UE ID range | Total rows | Size (MB) | Simulated time | Role |
|---|---:|---:|---:|---:|---:|---|
| part1 | 119 | 0--118 | 48,084,603 | 6,090 | 0--366 s | Main scenario 1 |
| part1.5 | 52 | 3--118 | 515,469 | 71 | 366--370 s | Transition window |
| part2 | 171 | 0--170 | 67,495,038 | 7,628 | 370--736 s | Main scenario 2 |
| part2.5 | 50 | 27--137 | 590,984 | 81 | 736--740 s | Transition window |
| part3 | 171 | 0--170 | 80,773,502 | 8,520 | 740--1,106 s | Main scenario 3 |
| part3.5 | 80 | 67--169 | 785,331 | 78 | 1,106--1,111 s | Transition window |
| part4 | 73 | 0--72 | 44,059,334 | 4,577 | 1,111--1,470 s | Main scenario 4 |
| **Total** | **716** | -- | **242,304,261** | **27,045** | -- | -- |

## 3. File Naming

Each file corresponds to one UE in one scenario part:

```text
<part>_ue_<id>_metrics.csv
```

The UE IDs in filenames are local to the corresponding simulation part. Restored
cross-part UE identity information is documented through the mapping files in
`metadata/`.

## 4. Schema

The released files follow a common event-log structure with a canonical schema of
up to 23 columns. Some files contain a subset of these columns when the
corresponding metrics were not emitted for that UE and scenario part.

### Temporal

| Column | Unit | Description |
|---|---|---|
| `time_s` | s | Simulation timestamp |

### Mobility / Spatial

| Column | Unit | Description |
|---|---|---|
| `vehicle_id` | -- | SUMO vehicle identifier |
| `x` | m | SUMO Cartesian X coordinate |
| `y` | m | SUMO Cartesian Y coordinate |
| `speed` | m/s | Vehicle speed |
| `edge` | -- | SUMO/OpenStreetMap road edge identifier |
| `latitude` | degrees | WGS84 latitude |
| `longitude` | degrees | WGS84 longitude |

### MAC / PHY

| Column | Unit | Description |
|---|---|---|
| `sinr_dl_db` | dB | Downlink scheduler SINR |
| `sinr_ul_db` | dB | Uplink scheduler SINR |
| `cqi_dl` | index | Downlink CQI |
| `cqi_ul` | index | Uplink CQI |
| `rcvd_sinr_dl_db` | dB | Downlink received SINR |
| `rcvd_sinr_ul_db` | dB | Uplink received SINR |

### RLC / Application

| Column | Unit | Description |
|---|---|---|
| `rlc_delay_dl_ms` | ms | Downlink RLC SDU delay |
| `rlc_delay_ul_ms` | ms | Uplink RLC SDU delay |
| `rlc_pdu_delay_dl_ms` | ms | Downlink RLC PDU delay |
| `rlc_pdu_delay_ul_ms` | ms | Uplink RLC PDU delay |
| `rlc_pdu_throughput_dl_bps` | bps | Downlink RLC PDU throughput |
| `rlc_pdu_throughput_ul_bps` | bps | Uplink RLC PDU throughput |
| `throughput_dl_bps` | bps | Downlink application-layer goodput |
| `throughput_ul_bps` | bps | Uplink application-layer goodput |
| `latency_ul_ms` | ms | Uplink end-to-end packet latency |

## 5. Event-Driven Metric Availability

The dataset is event-driven rather than sampled on a single dense time grid. A
row is created when a specific simulator event occurs, and only the metric
associated with that event is populated. For example, SUMO mobility events emit
position and speed, radio/channel events emit SINR and CQI, and packet delivery
events emit throughput or latency.

Empty cells therefore mean that the metric was not emitted at that event time.
They should not be interpreted as simulator failure or random missingness.

## 6. Metadata Files

- `metadata/gnodeb_opencellid_selected_19.csv`: selected 19 OpenCelliD-derived
  base-station records, augmented with simulation cell IDs, transmit powers, and
  antenna heights.
- `metadata/node_mapping_366_370.txt`: UE mapping for transition window part1.5.
- `metadata/node_mapping_736_740.txt`: UE mapping for transition window part2.5.
- `metadata/node_mapping_1106_1110.txt`: UE mapping for transition window part3.5.

## 7. Configuration Files

- `configs/omnetpp.ini`: OMNeT++/Simu5G scenario configuration, trimmed to the
  seven NeversNet5G scenario and transition configs used for this release.
- `configs/MultiCell_X2Mesh_19gNodeB_NR.ned`: 19-gNodeB NR topology with full X2 mesh.

## 8. Platform Notes

For Zenodo, upload the archives or folder contents directly. For Hugging Face
Datasets and Kaggle, split the data by scenario part to avoid very large single
uploads. GitHub should host only documentation, scripts, and small metadata
files; store the 26.4 GiB data payload in Zenodo, Hugging Face, or Kaggle.
