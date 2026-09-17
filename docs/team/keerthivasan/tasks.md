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

## Done — scaled to the full datasets

These were listed as "blocked on teammates" — resolved by validating the schema
myself at full scale rather than waiting, since the sample-scale version already
ran cleanly (see Thrishala's `review1_slide_notes.md`). Her judgment call on
whether the nearest-gNB heuristic is good enough still stands as the thing to
revisit; this only proves the *pipeline* scales, not that the *heuristic* is right.

- [x] Applied the node/edge schema to the **full 716-file / 27GB NeversNet5G
      dataset** — `build_full_ue_gnb_graph()` in `src/tgnn/build_graph.py`. Result:
      695 nodes (676 UEs + 19 gNBs), 676 edges, built in 10 minutes. Hit and fixed
      a real bug: pandas' C parser threw an out-of-memory `ParserError` on a
      handful of the larger UE files (~127MB); added a python-engine fallback so
      one bad file logs a skip instead of crashing the whole run (40/716 files
      skipped — no valid position rows, not a parsing failure).
- [x] Validated Milan at full scale — `src/data/milan_features.py`, aggregates all
      62 daily files (89.2M raw rows/day×cell×country → 89.2M aggregated
      (square_id, time_interval) rows). Confirms **exactly** 10,000 grid cells and
      8,928 unique 10-minute intervals (62 days × 144 intervals/day, no gaps) —
      the full dataset is clean and complete, not just the 1-week sample.
- [x] Ran Thrishala's TGNN baseline against the full-scale NeversNet5G graph
      (695 nodes): **56 milliseconds** for the forward pass — comfortably inside
      Krish's NFR1 control-loop latency budget (sub-10ms was the target for the
      *MARL* decision step; 56ms for TGNN inference on the full topology is a
      non-issue at this node count). Confirms `model.py` scales past the 29-node
      smoke test without any changes to the model itself.

## Done — the actual end-to-end closed loop

- [x] `src/digital_twin/orchestrator.py` — the first real, runnable closed loop:
      loads a real B5G sample (via Krish's `b5g_env.py`) → placeholder SSL
      embeddings shaped exactly like Sriranjana's real contract → Thrishala's
      `TGNNPredictor` → a heuristic stand-in for Krish's MADDPG action → the
      Digital Twin's approve/reject gate (`evaluate_in_twin`, FR4). Tested end to
      end against the real 416-node B5G graph: 389 real agents processed, 274
      approved, 115 rejected on SLA grounds. Every placeholder is explicitly
      marked and matches `interface_contracts.md` exactly, so swapping in each
      teammate's trained module later is a drop-in replacement, not a rewrite.
- [x] Found and fixed a real bug during integration: B5G's actual slice-type
      label is `mMTC`, not `mIoT` as every doc (including the literature review's
      dataset notes) assumed — corrected across `b5g_env.py`, `orchestrator.py`,
      and the affected docs. This is exactly the kind of mismatch integration
      testing is supposed to catch before review 1.

## Next (not yet started)

- [ ] Install OMNeT++ + Simu5G + SUMO integration so the scenario skeleton in
      `src/digital_twin/simu5g_scenario/` actually runs, replacing the offline
      B5G replay with a live network as the Network Element Layer.
- [ ] Wire up telemetry replay from the existing NeversNet5G CSVs as an
      alternative "Network Element Layer" data source (faster than a live
      simulator) once Thrishala's graph-construction work is validated.
- [ ] Replace the three placeholders in `orchestrator.py` (SSL embeddings, TGNN
      is real but untrained, MARL heuristic) with each teammate's trained module
      as they land — track this as the concrete "integration debt" list.
