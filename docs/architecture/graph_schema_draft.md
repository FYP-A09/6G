# Graph schema — draft (Keerthivasan, pending Thrishala's validation)

Thrishala's task is to define and validate the node/edge schema on the
NeversNet5G **sample** (`Thrishala_S_N_neversnet5g_sample.zip`, part1 + part1_5).
This is a draft built against the **full** dataset, prepared proactively since the
full data was already on hand — treat it as a starting proposal to react to, not a
finished spec. Once Thrishala's version is validated on the sample, reconcile the
two and this file should be updated or retired in favor of hers.

## NeversNet5G graph

- **Node types**:
  - `gNB` — 19 nodes, static position, from `metadata/gnodeb_opencellid_selected_19.csv` in the NeversNet5G release.
  - `UE` — one node per `vehicle_id`, position from `(x, y)` / `(latitude, longitude)`, time-varying.
- **Edges**:
  - `UE —serves→ gNB`: the UE's currently-serving cell at time `t`. Not directly
    labeled in the per-UE CSVs — has to be inferred from signal quality
    (`sinr_dl_db`/`rcvd_sinr_dl_db`) proximity to each gNB's known position, or from
    the X2-mesh handover behavior implied by the Simu5G config
    (`src/digital_twin/simu5g_scenario/MultiCell_X2Mesh_19gNodeB_NR.ned`).
  - `gNB —X2→ gNB`: static mesh links between gNodeBs, from the NED topology file —
    these don't change over time, unlike UE-gNB edges.
- **Edge weight candidates**: `sinr_dl_db` (signal-quality-weighted) or
  `rlc_pdu_throughput_dl_bps` (capacity-weighted) — pick based on which the TGNN's
  first ablation prefers.
- **Open question for Thrishala**: the release's own `DATA_DESCRIPTION.md` notes
  it "does not include the downstream edge-level filled graph used by the
  companion routing study" — meaning the UE-gNB serving-cell edge has to be
  reconstructed, not read directly. Validate the inference method on the sample
  before assuming it generalizes.

## B5G graph

- Already a real graph, no inference needed: `data/raw/b5g_slicing/2.2.1-10kprocessed/graphs/*.txt`
  are GML-format Topology Zoo network files (e.g. `Renater`, France backbone).
- **Node type**: network element (router/switch, per Topology Zoo convention).
- **Edge type**: physical link, direction and multigraph flags set in the GML header (`directed 1`, `multigraph 1`).
- Pairs with `2.2.1-10kprocessed/routings/` and `.../slices/` (per-flow eMBB/URLLC/mMTC
  QoS) for the MARL side — Krish's dataset, same topology files Thrishala needs for
  her generalization test (FR3 in her requirements).

## Milan grid ("graph" in the loose sense)

- Milan isn't graph-structured in the dataset itself — it's a 100×100 spatial grid
  (`grid_id` per row). Treat adjacent grid cells as edges (4- or 8-neighbor) if a
  graph structure is wanted here at all; otherwise use Milan purely as an SSL/TGNN
  *feature* source (per-cell traffic time series) rather than a topology source.
