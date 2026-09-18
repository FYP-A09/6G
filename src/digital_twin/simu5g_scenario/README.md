# Simu5G scenario skeleton (Network Element Layer)

This is the **actual** OMNeT++/Simu5G configuration used to generate the
NeversNet5G dataset (`data/raw/neversnet5g/`) — not a hand-written stand-in. Using
the real config means the Digital Twin's "Network Element Layer" (per the Native NDT
paper's 3-layer architecture) can be built against ground truth we already have full
telemetry for, instead of an invented topology.

## What's here

- `omnetpp.ini` — the trimmed release config. Kept blocks (per its own header
  comment): `SUMO_Integrator_Realistic_part_{1,2,3,4}` (the four main scenario
  windows) and `SUMO_Integrator_Realistic_{366,736,1106}` (the three transition
  windows) — these map 1:1 onto the `part1`–`part4` and `part1_5`–`part3_5` folders
  in `data/raw/neversnet5g/data/`.
- `MultiCell_X2Mesh_19gNodeB_NR.ned` — the network definition: 19 gNodeBs in an
  X2-mesh topology (matches the 19 real-world base-station locations described in
  the literature review's NeversNet5G entry), NR Standalone, over Nevers, France.

## Requirements to actually run this

Needs a local install of **OMNeT++ + Simu5G + veins/SUMO integration**. OMNeT++
6.4.0 + INET 4.7.0 + Simu5G 1.7.0 are now installed and validated on this machine
(WSL2 Ubuntu, via `opp_env` in nixless mode — see
`docs/team/keerthivasan/tasks.md` for the exact build steps and fixes, and
`docs/figures/simu5g_nr_validation.png` for a real simulation run through the
installed toolchain: 1 gNB + 1 UE, 5s, 28,092 events, MOS 4.41/5, 0 packet loss).
**veins/SUMO is not installed yet** — this specific `MultiCell_X2Mesh_19gNodeB_NR.ned`
scenario (19 gNodeBs + SUMO-driven vehicle mobility over Nevers, France) still
needs that integration wired in and its own scenario-specific debugging before it
runs end to end; the toolchain being installed removes the *simulator* blocker but
not this remaining scenario-integration work. `**.numUe` in each config block (e.g.
119 for part_1) shows the original UE count per scenario window; scale this down
for a faster local smoke test before attempting a full 1,470s replay.

## How this fits the project

This scenario is the "physical network" half of the Digital Twin loop: the NDT
Layer (SSL → TGNN → MARL, see `docs/architecture/`) needs a live source of
topology + telemetry to synchronize against, either by (a) replaying the existing
NeversNet5G CSVs as if they were live telemetry (fast to wire up, no simulator
install needed — good for the first end-to-end demo), or (b) actually re-running
this Simu5G config with modified slice/traffic parameters to generate *new*
scenarios beyond what's in the static dataset (needed once policies must be tested
against conditions the original dataset didn't cover).

Start with (a) for review 1; move to (b) once the closed loop needs live variation.
