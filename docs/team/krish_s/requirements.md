# Requirements — MARL / Dynamic Slicing (Krish S)

## Functional
- **FR1** Agents (one per gNB/edge site) decide per-slice resource allocation (PRB share / bandwidth) at each control interval.
- **FR2** Support all three 3GPP slice types (eMBB, URLLC, mMTC) with distinct QoS targets (latency, throughput, reliability).
- **FR3** Reward penalizes SLA violations (latency over budget, packet loss over threshold) *and* reconfiguration churn (allocating too often).
- **FR4** Agent state includes the TGNN's predicted per-slice demand, not just the current observation — this is the reactive→predictive shift that is the project's core novelty claim.
- **FR5** Policies train/validate inside the Digital Twin sandbox before being considered for physical rollout.

## Non-functional
- **NFR1** Decision latency budget: inference must fit inside the URLLC control interval. The vehicular-network MADDPG paper's 120–180ms coordination delay violated URLLC — beat that.
- **NFR2** Scalability beyond a single tightly-localized cell cluster (the gap identified against Li et al.'s SF-DTMC paper).
- **NFR3** Use a standard Gym/PettingZoo API so results are comparable against published baselines (MADDPG: 96.5% QoS satisfaction; Safety DRL: <1% SLA violations).

## Data requirements
- Per-flow/per-slice QoS (delay, jitter, loss) plus topology and provisioning state → **B5G Network Slicing Dataset** (`data/raw/b5g_slicing/`), which already includes deliberately over/under-provisioned scenarios as negative examples for the reward function.
- A ready reward proxy for early prototyping, before the full reward function is tuned → **6G RAN Telemetry for FL** (`data/raw/6g_ran_telemetry_fl/`), which already ships computed `reward` and `sla_ok_next` columns per client-window.
