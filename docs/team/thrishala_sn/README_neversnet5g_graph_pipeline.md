# NeversNet5G → Graph Sequence: Data Processing (TGNN stage)

**Owner:** TGNN / Traffic & Topology Prediction
**Code:** `src/tgnn/build_graph.py`
**Input:** `data/raw/neversnet5g/`, `data/raw/neversnet5g_metadata/`, `data/raw/b5g_slicing(_sample)/`
**Output:** `data/processed/neversnet5g_graphs/<part_label>/` (one folder per processed part/run)

---

## 1. What this solves

NeversNet5G's per-UE CSVs are **event-driven** (see `DATA_DESCRIPTION.md` §5): a
row only exists when some simulator event fires, and only the columns tied to
that event are populated. There is no dense per-UE time grid, no per-UE dense
position trace, and no labeled serving cell. That's unusable directly by a TGNN,
which needs a **regular sequence of graph snapshots** — same node/edge schema,
evenly spaced in time, every node's features known (or explicitly marked
unknown) at every step.

This pipeline turns the raw event logs into exactly that.

## 2. Pipeline, step by step

```
raw event log (per UE, per part)
        │
        ▼
1. Coalesce duplicate timestamps
   (different metric families can log the same time_s in separate rows —
   one row per timestamp, last non-null value per column wins)
        │
        ▼
2. Resample onto a fixed time grid
   (merge_asof, direction="backward" = last-observation-carried-forward:
   "what did we last know about this UE as of bin start T")
   → bins before the UE's first-ever event are DROPPED, not filled with a guess
        │
        ▼
3. Re-derive serving gNB, per bin
   (nearest-gNB-by-distance heuristic, re-evaluated at each bin's
   forward-filled position — so handovers show up as the edge target changing
   across the sequence)
        │
        ▼
4. Assemble one nx.Graph per bin
   (gNBs = always-present static nodes; UEs = present only from their first
   event onward; edges = UE → serving gNB, weighted by forward-filled SINR)
        │
        ▼
5. (optional) Stitch consecutive parts across a transition window
   using node_mapping_*.txt, so UE identity survives the part1→part1.5
   boundary instead of resetting
        │
        ▼
6. Write snapshot_<i>.gml per bin + manifest.json (coverage stats included)
```

## 3. Parameters — what to tune and why

| Parameter | Default | What it controls | Tuning guidance |
|---|---|---|---|
| `--bin-size-s` | `10.0` | Width of each time bin (temporal resolution of the output sequence) | **The most important knob.** Smaller = finer resolution but more bins are "stale" (carried-forward, not fresh) since events are sparse. Larger = every bin has fresher data but coarser resolution and faster handovers get missed. Start at 10s (matches nothing in particular — just a reasonable first guess given ~2–12s inter-event gaps observed on the sample); re-tune once `mean_known_sinr_fraction` from the manifest is checked (see §5). |
| `--max-ues` | `None` (all) | Caps UE files processed, for fast local iteration | Use e.g. `10` while developing, remove for the real run once the pipeline is validated. |
| `--part-dir` | `data/raw/neversnet5g/data/part1` | Which scenario part to process | One run per main part (`part1`..`part4`); each is processed independently unless stitched (see below). |
| `--transition-dir` + `--node-mapping` | `None` | Stitches the following transition window (e.g. `part1_5`) onto `--part-dir`, relabeling UE ids via the mapping file | Only set both together — a transition dir without its mapping file can't be relabeled and would silently create duplicate/disconnected UE identities. |
| `--gnb-metadata` | `data/raw/neversnet5g_metadata/gnodeb_opencellid_selected_19.csv` | gNB positions/range/tx-power source | Shouldn't normally change — one metadata file covers all parts. |
| `--b5g-max-graphs` | `5` | How many B5G topologies to load for the FR3 generalization check | Raise once more sample topologies are available; keep low while iterating locally. |
| `--out-dir` | `data/processed/neversnet5g_graphs/part1` | Where snapshots + manifest are written | One directory per run — don't reuse a directory across different `--bin-size-s` values, the manifest won't tell you which snapshot came from which setting otherwise. |

## 4. Output schema

Each `snapshot_<i>.gml` is one `networkx.Graph`:

**gNB node** (`type="gnb"`, always present, every snapshot):
`lat`, `lon`, `tx_power_dbm`, `antenna_height_m`, `bs_id`

**UE node** (`type="ue"`, present only once that UE has ≥1 event):
`lat`, `lon`, `vehicle_id`, plus **whichever of** `sinr_dl_db`, `sinr_ul_db`,
`cqi_dl`, `cqi_ul`, `throughput_dl_bps`, `throughput_ul_bps`, `latency_ul_ms`,
`speed` **were known as of that bin** (forward-filled from the most recent
event). **A feature not yet observed for that UE is OMITTED from the node's
attribute dict — not written as `None`/`NaN`/0.** GML has no native null, so
writing `None` would round-trip as the literal string `"None"`, silently
misread as real data downstream. Always read with `.get(key)`, never `[key]`.

**UE→gNB edge**: `weight` (= `sinr_dl_db` if known, else `0.0` — **a `weight`
of exactly `0.0` is therefore ambiguous between "measured 0 dB" and "unknown";
if that distinction matters, check the node's own `sinr_dl_db` key instead of
the edge weight**), `distance_m`, `in_range` (1/0 — whether the UE was within
that gNB's OpenCellID `range_m` when this association was made).

`manifest.json` per output directory: bin start times (so snapshot index →
real simulated-time seconds is recoverable), the parameters used, and a
coverage report (§5).

## 5. Reading the coverage report before trusting the output

`summarize_sequence()` (also printed by the CLI and saved into `manifest.json`)
reports, across the whole sequence:

- `ue_count_per_bin_min/max/mean` — how many UEs are actually present per
  snapshot. If this is much lower than the part's total UE count, most UEs
  haven't had their first event yet at that point, or the run used
  `--max-ues`.
- `mean_in_range_fraction` — of all UE↔gNB associations made, what fraction
  were within the gNB's reported coverage `range_m`. Low values suggest either
  sparse gNB placement relative to UE positions, or that `bin_size_s` is wide
  enough that positions are stale by the time they're used.
- `mean_known_sinr_fraction` — of all UE↔gNB edges, what fraction had a
  **genuinely fresh-or-recent** SINR reading rather than defaulting to `0.0`
  because none had ever been observed yet. **This is the number to watch when
  tuning `--bin-size-s`** — if it's very low, radio events are sparse enough
  relative to the bin width that most edges are flying blind on signal
  quality, and either the bin size needs to grow or the model needs to treat
  `sinr` as missing-often and lean more on structural features.

On the small synthetic fixture used to validate this code (5 UEs, 10s bins),
`mean_known_sinr_fraction` came out at ~3% — expected, since radio events are
one of three roughly-equally-likely event types in the fixture and 10s bins
are wide relative to the ~2–12s synthetic inter-event gap. **Re-check this
number on the real sample** (`part1` + `part1_5`) before picking a final
`bin_size_s` — the real inter-event timing may differ substantially from the
synthetic fixture used here.

## 6. Known limitations (open items, not silently swept under the rug)

- **Serving-cell is a geometric heuristic, not ground truth.** `nearest_gnb`
  ignores handover hysteresis, building penetration loss, and doesn't use the
  UE's own measured SINR to pick the cell. Per `tasks.md`, this needs
  validating against the real sample (spot-check a handful of UEs for
  geographic sanity) before treating it as reliable at scale.
- **Cross-part UE identity only exists at transition boundaries.** Two UEs in,
  say, `part1` and `part2` that are actually the same vehicle have no direct
  link unless traced through the intervening transition window's
  `node_mapping_*.txt`. `stitch_part_and_transition` handles one hop
  (main → its following transition); chaining across multiple main parts
  would need calling it twice in sequence and hasn't been tested end-to-end
  yet.
- **Edge weight defaulting to `0.0` for "unknown"** is a modeling shortcut,
  not a clean missing-value representation — see the schema note in §4. Worth
  revisiting with Sriranjana/Krish if it causes the model to treat "no signal
  measured yet" as "measured exactly 0 dB."
- **B5G's per-scenario `slices/` and `routings/` subfolders aren't touched by
  this file** — only `graphs/` (topology) is loaded here. If MARL (Krish) or
  the DT (Keerthivasan) need those for QoS/slice ground truth, that's a
  separate loader.
- **Not yet validated against the real 6.4GB sample** — everything above was
  validated end-to-end against a synthetic fixture (`make_fixture.py`,
  included for reference) that mimics the real schema and event-driven
  sparsity pattern, since the real sample archive wasn't available in this
  environment. Re-run against the real `part1`/`part1_5` sample before
  treating the coverage numbers as representative.

## 7. How to run

```bash
# Single part, no stitching
python src/tgnn/build_graph.py \
  --part-dir data/raw/neversnet5g/data/part1 \
  --bin-size-s 10 \
  --out-dir data/processed/neversnet5g_graphs/part1

# Part1 stitched onto its transition window (continuous UE identity across the boundary)
python src/tgnn/build_graph.py \
  --part-dir data/raw/neversnet5g/data/part1 \
  --transition-dir data/raw/neversnet5g/data/part1_5 \
  --node-mapping data/raw/neversnet5g_metadata/node_mapping_366_370.txt \
  --bin-size-s 10 \
  --out-dir data/processed/neversnet5g_graphs/part1_stitched

# Fast local iteration (10 UEs only)
python src/tgnn/build_graph.py --part-dir data/raw/neversnet5g/data/part1 --max-ues 10 \
  --out-dir data/processed/neversnet5g_graphs/part1_smoketest
```

Each run also loads and reports on the B5G topologies (`--b5g-max-graphs`,
default 5) and prints a train/holdout topology split for the FR3
generalization test — this doesn't touch NeversNet5G output, just runs
alongside it as a convenience.

## 8. Handoff — what the next stages get from this

- **To Sriranjana (SSL):** the raw per-node feature values in each snapshot
  (`sinr_dl_db`, `throughput_dl_bps`, etc.) are exactly the "unlabeled
  telemetry" her pretext tasks pretrain on — read snapshots in sequence per
  node to build the windows her encoder needs.
- **To this module itself (TGNN, `model.py`):** once Sriranjana's encoder
  produces real per-node embeddings, replace `model.py`'s dummy
  `torch.randn(n_nodes, n_timesteps, 64)` with embeddings indexed against
  these same node ids and the same bin sequence, and derive the adjacency
  matrix per-timestep from each snapshot's edges (currently `model.py` assumes
  one static `adj` for the whole window — that assumption needs revisiting
  once handovers are visible in the data, since the serving gNB can change
  bin-to-bin).
- **To Krish (MARL):** node/edge `in_range` and `distance_m` are available as
  extra structural signal if useful for reward shaping around coverage
  quality, beyond just the QoS metrics coming from B5G Slicing.

## 9. Files in this delivery

| File | Purpose |
|---|---|
| `build_graph.py` | The pipeline itself — drop into `src/tgnn/build_graph.py`, replacing the previous version. |
| `make_fixture.py` | Generates a small synthetic event-driven dataset matching the real schema, used to validate the pipeline in an environment without the real 6.4GB sample. Not part of the production pipeline — safe to delete once validated against real data, or keep as a lightweight CI smoke test. |
| `README.md` | This file. |
