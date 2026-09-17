# Module interface contracts

Owned by Keerthivasan (Digital Twin + Integration) — this is the fixed spec each
module builds against, so Sriranjana/Thrishala/Krish don't have to guess at each
other's output shape. Grounded in the actual columns present in the datasets each
module owns (see `data/raw/README.md`), not placeholder fields.

## 1. SSL → TGNN: node embedding

Sriranjana's module emits one embedding per (node, timestep). "Node" means a UE or
gNB for NeversNet5G-shaped data, or a grid cell for Milan-shaped data.

| Field | Type | Notes |
|---|---|---|
| `node_id` | string | `vehicle_id` (NeversNet5G) or `grid_id` (Milan) |
| `timestamp` | float (s) | `time_s` (NeversNet5G) or the 10-minute interval start, as epoch ms (Milan) |
| `embedding` | float[D] | Fixed dimension `D`, agreed at **D=64** as the starting point (revisit after the first ablation) |
| `source_dataset` | enum | `neversnet5g` \| `milan` \| `b5g` — so the TGNN encoder knows which raw-feature adapter produced it |

Raw fields the SSL encoder consumes before producing the embedding (for reference,
so Sriranjana's encoder input side is unambiguous too):
- **NeversNet5G**: `sinr_dl_db`, `sinr_ul_db`, `cqi_dl`, `cqi_ul`, `rlc_pdu_throughput_dl_bps`, `rlc_pdu_throughput_ul_bps`, `x`, `y`, `speed`.
- **Milan**: `sms_in`, `sms_out`, `call_in`, `call_out`, `internet` (the 5 traffic columns in each aggregated `CellID`/`datetime` row; the current sample source fields are `smsin`, `smsout`, `callin`, `callout`, `internet` in `data/raw/milan_telecom_italia/*.csv`).

The validated export path is `src/ssl/tgnn_bridge.py`, which orders these
records as `[N, T, 64]` for the TGNN while preserving `node_id`, `timestamp`,
and `source_dataset` metadata.

## 2. TGNN → MARL: predicted demand

Thrishala's module emits a predicted-demand vector per (node or slice, future
timestep window), consumed as part of MARL agent state.

| Field | Type | Notes |
|---|---|---|
| `entity_id` | string | Node id (fine-grained) or slice id (aggregated) — decide grain per experiment |
| `slice_type` | enum | `eMBB` \| `URLLC` \| `mMTC` |
| `horizon_start` / `horizon_end` | float (s) | The future window this prediction covers |
| `predicted_throughput_bps` | float | |
| `predicted_latency_ms` | float | |
| `predicted_load` | float, 0-1 | Normalized utilization, for the MARL reward's utilization term |
| `confidence` | float, 0-1 | Optional but recommended — lets MARL discount low-confidence predictions early in training |

## 3. MARL → NDT: candidate action

Krish's module emits a candidate resource-allocation action, which the Digital Twin
evaluates *before* it's allowed to reach the "Application Layer" (physical
rollout candidate).

| Field | Type | Notes |
|---|---|---|
| `agent_id` | string | gNB or edge-site id issuing the action |
| `slice_allocations` | dict\<slice_type, float\> | PRB or bandwidth share per slice, sums to ≤1.0 |
| `predicted_reward` | float | The agent's own estimate, for logging/debugging drift between agent belief and twin-measured outcome |
| `twin_evaluation` | enum | `pending` \| `approved` \| `rejected` — set by the Digital Twin, not the agent |
| `rejection_reason` | string, optional | Filled in when `twin_evaluation = rejected` (e.g. "predicted SLA violation on URLLC slice") |

## Versioning

Any change to a field name, type, or the embedding dimension `D` is a breaking
change — bump a `schema_version` string alongside these contracts once code starts
depending on them, so a stale consumer fails loudly instead of silently
misreading a field.
