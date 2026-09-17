# Tasks — Digital Twin + Integration (Keerthivasan)

**Data split:** you hold the full NeversNet5G (28GB) and full Milan (20GB) on
`E:\FYP DATA\6G\data\raw\` — too large to hand to teammates as zips, so you own
scaling their sample-validated designs up to the full data instead of transferring
it. Thrishala prototypes against a NeversNet5G part-folder sample and a Milan
1-week sample; once her graph schema and TGNN baseline are validated on those, apply
the same logic to the full datasets here and report back whether it holds up at
scale (more nodes, more days, real long-range temporal patterns).

## Done

- [x] Finalize and circulate the module interface contracts — see
      [`docs/architecture/interface_contracts.md`](../../architecture/interface_contracts.md)
      (SSL→TGNN embedding format, TGNN→MARL prediction format, MARL→NDT action
      format, all grounded in the actual dataset columns rather than placeholders).
- [x] Stand up a first Simu5G scenario skeleton — see
      [`src/digital_twin/simu5g_scenario/`](../../../src/digital_twin/simu5g_scenario/).
      This is the **actual** config that generated the NeversNet5G dataset (19-gNodeB
      X2-mesh over Nevers, France), not an invented stand-in. Not runnable yet
      (OMNeT++/Simu5G/SUMO aren't installed on this machine) — that install is the
      next concrete step here.
- [x] Draft the end-to-end architecture diagram — see
      [`docs/architecture/system_diagram.md`](../../architecture/system_diagram.md).
- [x] Draft a candidate graph node/edge schema for NeversNet5G/B5G/Milan ahead of
      Thrishala's validation pass — see
      [`docs/architecture/graph_schema_draft.md`](../../architecture/graph_schema_draft.md).
      Explicitly a draft: reconcile with her sample-validated version once she has one.
- [x] Keep `data/raw/README.md` and `src/data/download_datasets.py` current —
      up to date as of the full-vs-sample dataset split.
- [x] Prepare the review-1 slide — see
      [`review1_slide_notes.md`](review1_slide_notes.md).

## Blocked on teammates (can't be finished solo)

- [ ] Apply Thrishala's node/edge schema to the full 8-part NeversNet5G dataset
      once she's validated it on the sample — genuinely depends on her output, not
      just data access. The draft schema above is a starting point, not a
      substitute for her validation.
- [ ] Re-run her GraphSAGE + temporal-conv baseline against full Milan/NeversNet5G
      once it works on the samples — same dependency.

## Next (not yet started)

- [ ] Install OMNeT++ + Simu5G + SUMO integration so the scenario skeleton in
      `src/digital_twin/simu5g_scenario/` actually runs.
- [ ] Wire up telemetry replay from the existing NeversNet5G CSVs as the first
      "Network Element Layer" data source (faster than a live simulator — see the
      scenario README's option (a)) so the NDT loop has something to synchronize
      against for a first end-to-end demo.
