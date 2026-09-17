# Design — Digital Twin + Integration (Keerthivasan)

## Three-layer architecture (per the Native NDT paper)
1. **Network Element Layer** — physical/simulated gNBs, UEs, UPFs.
2. **NDT Layer** — this project's SSL + TGNN + MARL intelligence core.
3. **Application Layer** — orchestration decisions: PRB allocation, VNF
   scaling/migration.

## Data contracts between modules
- **SSL → TGNN**: fixed-size node embedding, versioned schema (owned jointly with
  Sriranjana).
- **TGNN → MARL**: per-slice demand prediction vector, timestamped (owned jointly
  with Thrishala and Krish).
- **MARL → NDT**: candidate action, evaluated in the twin before promotion to an
  "Application Layer" decision.

## Gray-box modeling
Per the Native NDT paper: combine protocol/domain knowledge (3GPP slice QoS
definitions, known topology constraints) with the learned SSL/TGNN/MARL components,
rather than treating the whole pipeline as a black box.

## Repo/infra design (already implemented)
- `data/raw/` catalog with a download function per dataset; large raw data
  gitignored; small labeled datasets committed directly for team convenience.
- `docs/` for literature review, progress notes, and per-person requirements/design/
  tasks.

## Still to design
The actual simulator backbone (Simu5G or ns-3 5G-LENA) that stands in for the
"Network Element Layer" once real-time synchronization needs to be demonstrated —
no existing dataset provides a live closed loop, so this has to be built rather than
sourced.
