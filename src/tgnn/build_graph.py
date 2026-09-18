"""
Graph construction for the TGNN stage (Thrishala S N's task — see
docs/team/thrishala_sn/tasks.md, design.md, and docs/architecture/graph_schema_draft.md).

Two topology sources, per the draft schema:

1. NeversNet5G: UE/gNB graph. gNB nodes + positions/range/tx-power come from the
   metadata CSV. The per-UE CSVs are EVENT-DRIVEN (see DATA_DESCRIPTION.md §5) —
   a row exists only when some simulator event fires, and only the columns tied to
   that event are populated. There is no dense per-UE time grid and no per-UE
   dense position/SINR trace. This module turns that sparse event log into a
   regular sequence of graph snapshots by:
     (a) coalescing duplicate/near-duplicate event timestamps per UE,
     (b) resampling onto a fixed-width time grid via last-observation-carried-
         forward (merge_asof, backward direction) — i.e. "what did we last know
         about this UE as of bin start T",
     (c) re-deriving the UE's serving gNB *per bin* via nearest-gNB-by-distance
         (a heuristic — the release does not label serving cell directly; see
         nearest_gnb's docstring for the open question this leaves).
   UE identity is only local within one scenario part. Cross-part identity is
   only recoverable at the transition windows (part1.5/2.5/3.5) via the
   node_mapping_*.txt files (original_id -> sequential_id) — see
   `load_node_mapping` and `stitch_part_and_transition`.

2. B5G: already a real graph (GML files) — loaded directly, no inference needed.
   Extended here to load several scenario graphs for the FR3 generalization test
   (train/evaluate across more than one topology).

Requires: networkx, pandas. No torch needed for this file.
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
import re
from dataclasses import dataclass, field

import networkx as nx
import pandas as pd

_DATA_RAW = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
_DATA_PROCESSED = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")

GNB_METADATA_CSV = os.path.join(
    _DATA_RAW, "neversnet5g_metadata", "gnodeb_opencellid_selected_19.csv"
)

# All non-identifier, non-time columns the per-UE CSVs may contain (schema §4 of
# DATA_DESCRIPTION.md). Carried forward together as one row per merge_asof call —
# no per-column special-casing needed, missing means "not yet emitted".
UE_METRIC_COLUMNS = [
    "x", "y", "speed", "edge", "latitude", "longitude",
    "sinr_dl_db", "sinr_ul_db", "cqi_dl", "cqi_ul",
    "rcvd_sinr_dl_db", "rcvd_sinr_ul_db",
    "rlc_delay_dl_ms", "rlc_delay_ul_ms",
    "rlc_pdu_delay_dl_ms", "rlc_pdu_delay_ul_ms",
    "rlc_pdu_throughput_dl_bps", "rlc_pdu_throughput_ul_bps",
    "throughput_dl_bps", "throughput_ul_bps", "latency_ul_ms",
]

# Columns actually used as node/edge features downstream (model.py / SSL input).
# Kept as a separate, smaller list from UE_METRIC_COLUMNS so it's easy to extend
# the raw schema without silently changing what the model sees.
NODE_FEATURE_COLUMNS = [
    "sinr_dl_db", "sinr_ul_db", "cqi_dl", "cqi_ul",
    "throughput_dl_bps", "throughput_ul_bps", "latency_ul_ms", "speed",
]

# Targets are kept separate from SSL inputs because this sample has sparse
# application throughput fields. The RLC UL fields are the populated proxy
# used by the NeversNet5G TGNN trainer when the advertised DL fields are empty.
TARGET_FEATURE_COLUMNS = [
    "rlc_pdu_throughput_ul_bps", "rlc_pdu_delay_ul_ms",
]

_UE_FILENAME_RE = re.compile(
    r"(?P<part>part\d+(?:[._]5)?)_ue_(?P<ue_id>\d+)_metrics\.csv$"
)

# =====================================================================
# gNB metadata
# =====================================================================

@dataclass
class GnbInfo:
    gnb_index: int
    bs_id: str
    lat: float
    lon: float
    range_m: float
    tx_power_dbm: float
    antenna_height_m: float


def load_gnb_metadata(metadata_csv: str = GNB_METADATA_CSV) -> dict[int, GnbInfo]:
    """Returns {gnb_index: GnbInfo} from the NeversNet5G release's gNB metadata.

    Note the CSV's column order is (lon, lat) even though we read by header name,
    so this is safe regardless of column order.
    """
    gnbs: dict[int, GnbInfo] = {}
    with open(metadata_csv, newline="") as f:
        for row in csv.DictReader(f):
            idx = int(row["gnb_index"])
            gnbs[idx] = GnbInfo(
                gnb_index=idx,
                bs_id=row["bs_id"],
                lat=float(row["lat"]),
                lon=float(row["lon"]),
                range_m=float(row["range_m"]),
                tx_power_dbm=float(row["tx_power_dbm"]),
                antenna_height_m=float(row["antenna_height_m"]),
            )
    return gnbs


def load_node_mapping(mapping_txt: str) -> dict[int, int]:
    """Parses a node_mapping_<t0>_<t1>.txt transition file into
    {original_id (main-part-local UE id): sequential_id (transition-part-local UE id)}.

    Only exists for the transition windows (part1.5/2.5/3.5); this is the ONLY
    place cross-part UE identity can be recovered — the main parts do not share
    UE id spaces with each other directly.
    """
    mapping: dict[int, int] = {}
    with open(mapping_txt) as f:
        reader = csv.DictReader(line for line in f if not line.startswith("#"))
        for row in reader:
            mapping[int(row["original_id"])] = int(row["sequential_id"])
    return mapping


# =====================================================================
# Per-UE event log -> regular time-binned trace
# =====================================================================

def discover_ue_files(part_dir: str) -> list[tuple[int, str]]:
    """Returns [(ue_id, filepath), ...] for every `<part>_ue_<id>_metrics.csv`
    in part_dir, sorted by ue_id."""
    files = []
    for path in glob.glob(os.path.join(part_dir, "*_ue_*_metrics.csv")):
        m = _UE_FILENAME_RE.search(os.path.basename(path))
        if m:
            files.append((int(m.group("ue_id")), path))
    return sorted(files)


def _read_ue_csv(csv_path: str, usecols: list[str]) -> pd.DataFrame | None:
    """Reads one UE CSV, tolerating the failure modes actually seen across the
    full 716-file/27GB release. Returns None (caller treats as empty) rather
    than raising, so one bad file doesn't abort a full-scale build.

    A handful of the larger files (~127MB) have hit the C parser's tokenizer
    with an "out of memory" ParserError on this machine — the file itself
    isn't corrupt, the default parser buffer just chokes on it. Retry with
    the slower pure-Python engine before giving up on the file.
    """
    try:
        return pd.read_csv(csv_path, usecols=lambda c: c in usecols, low_memory=False)
    except ValueError:
        # A subset of files omit part of the canonical schema (DATA_DESCRIPTION.md
        # §4) in a way the usecols callable rejects — fall back to reading
        # whatever columns are actually present.
        try:
            return pd.read_csv(csv_path, low_memory=False)
        except Exception as exc:
            print(f"  [skip] {os.path.basename(csv_path)}: unreadable ({exc})")
            return None
    except Exception as exc:
        try:
            return pd.read_csv(csv_path, usecols=lambda c: c in usecols, engine="python")
        except Exception:
            print(f"  [skip] {os.path.basename(csv_path)}: unreadable even with the "
                  f"python engine ({exc})")
            return None


def _load_and_coalesce_events(csv_path: str) -> pd.DataFrame:
    """Loads one UE's event log and coalesces rows that share a timestamp.

    The dataset is event-driven (DATA_DESCRIPTION.md §5): different metric
    families are emitted independently, so two rows can carry the same
    `time_s` with disjoint sets of populated columns (e.g. one mobility event,
    one radio event, logged at the same instant). Coalescing keeps one row per
    timestamp, taking the last non-null value per column at that timestamp.

    That alone only merges rows that share the *exact* timestamp, though — a
    column populated by an earlier event and left untouched by a later,
    different-column event would otherwise still read back as NaN at the
    later timestamp. The trailing `.ffill()` carries each column's last known
    value forward across *all* prior timestamps (true last-observation-
    carried-forward), so a subsequent merge_asof sees one coherent, fully
    forward-filled observation per time point rather than a sparse one.
    """
    usecols = ["time_s", "vehicle_id"] + UE_METRIC_COLUMNS
    df = _read_ue_csv(csv_path, usecols)
    if df is None or "time_s" not in df.columns or df.empty:
        return pd.DataFrame(columns=usecols).iloc[0:0]
    df = df.sort_values("time_s")
    # last non-null per column, per timestamp
    coalesced = df.groupby("time_s", as_index=False).last()
    coalesced = coalesced.sort_values("time_s").reset_index(drop=True)
    fill_columns = [c for c in UE_METRIC_COLUMNS if c in coalesced.columns]
    coalesced[fill_columns] = coalesced[fill_columns].ffill()
    return coalesced


def resample_ue_to_bins(
    csv_path: str, bin_starts: "pd.Series | list[float]"
) -> pd.DataFrame:
    """Resamples one UE's sparse event log onto a fixed grid of bin-start times,
    using last-observation-carried-forward (merge_asof, direction='backward').

    Returns a DataFrame indexed by `time_s` == bin start, one row per bin, with
    every column in UE_METRIC_COLUMNS forward-filled from the most recent event
    at or before that bin start. Bins before the UE's first event are dropped —
    the UE is treated as "not yet present" rather than filled with a guess.
    """
    events = _load_and_coalesce_events(csv_path)
    if events.empty:
        return events

    bins_df = pd.DataFrame({"time_s": sorted(set(bin_starts))})
    merged = pd.merge_asof(
        bins_df, events, on="time_s", direction="backward", allow_exact_matches=True
    )
    first_event_t = events["time_s"].iloc[0]
    merged = merged[merged["time_s"] >= first_event_t].reset_index(drop=True)
    return merged


# =====================================================================
# gNB association
# =====================================================================

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters — good enough at city scale (Nevers, France)."""
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def nearest_gnb(lat: float, lon: float, gnbs: dict[int, GnbInfo]) -> tuple[int, float, bool]:
    """Nearest-gNB-by-distance heuristic — the reconstruction method flagged as an
    open question in graph_schema_draft.md (the dataset does not label serving
    cell directly). Ignores handover hysteresis and building penetration loss,
    and does NOT use the measured SINR to pick the cell — it is a geometric
    approximation, not ground truth.

    Returns (gnb_index, distance_m, in_range) where in_range is whether the UE
    falls within that gNB's reported OpenCellID range_m — a cheap sanity signal
    for how much to trust the association at that distance.
    """
    best_id, best_dist = min(
        ((gid, _haversine_m(lat, lon, g.lat, g.lon)) for gid, g in gnbs.items()),
        key=lambda pair: pair[1],
    )
    in_range = best_dist <= gnbs[best_id].range_m
    return best_id, best_dist, in_range


# =====================================================================
# Snapshot / sequence construction
# =====================================================================

def make_bin_edges(t_start: float, t_end: float, bin_size_s: float) -> list[float]:
    """Bin start times covering [t_start, t_end), step bin_size_s."""
    n_bins = max(1, math.ceil((t_end - t_start) / bin_size_s))
    return [t_start + i * bin_size_s for i in range(n_bins)]


def build_temporal_graph_sequence(
    part_dir: str,
    gnbs: dict[int, GnbInfo],
    bin_size_s: float = 10.0,
    t_start: float | None = None,
    t_end: float | None = None,
    max_ues: int | None = None,
    part_label: str | None = None,
) -> tuple[list[float], dict[float, nx.Graph]]:
    """Builds one graph snapshot per time bin for a NeversNet5G scenario part.

    Node set per snapshot: all gNBs (always present, static) + every UE that has
    at least one event at or before that bin (present-since-first-event; UEs with
    no events yet in this bin are simply absent from that snapshot's node set,
    not zero-filled).
    Edge set per snapshot: one UE -> serving-gNB edge per present UE, from
    `nearest_gnb` re-evaluated at that bin's forward-filled position, so
    handovers across bins show up as the edge changing target.

    bin_size_s is the single most important knob here — it trades off temporal
    resolution against how much of each bin is genuinely fresh data versus
    carried-forward staleness (see the coverage stats this function's caller
    should log — `summarize_sequence` below).
    """
    ue_files = discover_ue_files(part_dir)
    if max_ues:
        ue_files = ue_files[:max_ues]
    if not ue_files:
        raise FileNotFoundError(f"No *_ue_*_metrics.csv files found under {part_dir}")

    if part_label is None:
        m = _UE_FILENAME_RE.search(os.path.basename(ue_files[0][1]))
        part_label = m.group("part") if m else os.path.basename(part_dir)

    # First pass: find the part's actual time span if not given, so bins aren't
    # wastefully built outside the data (or clipped inside it).
    if t_start is None or t_end is None:
        obs_min, obs_max = math.inf, -math.inf
        for _, path in ue_files:
            ev = _load_and_coalesce_events(path)
            if ev.empty:
                continue
            obs_min = min(obs_min, float(ev["time_s"].iloc[0]))
            obs_max = max(obs_max, float(ev["time_s"].iloc[-1]))
        t_start = t_start if t_start is not None else obs_min
        t_end = t_end if t_end is not None else obs_max

    bin_starts = make_bin_edges(t_start, t_end, bin_size_s)

    snapshots: dict[float, nx.Graph] = {t: nx.Graph() for t in bin_starts}
    for t in bin_starts:
        for gid, g in gnbs.items():
            snapshots[t].add_node(
                f"gnb_{gid}", type="gnb", lat=g.lat, lon=g.lon,
                tx_power_dbm=g.tx_power_dbm, antenna_height_m=g.antenna_height_m,
                bs_id=g.bs_id,
            )

    for ue_id, path in ue_files:
        binned = resample_ue_to_bins(path, bin_starts)
        if binned.empty:
            continue
        for _, row in binned.iterrows():
            t = float(row["time_s"])
            lat, lon = row.get("latitude"), row.get("longitude")
            if pd.isna(lat) or pd.isna(lon):
                continue  # no position known yet at this bin — UE absent, not zero-filled
            gnb_id, dist_m, in_range = nearest_gnb(float(lat), float(lon), gnbs)

            node_id = f"ue_{part_label}_{ue_id}"
            features = {
                col: float(row[col])
                for col in [*NODE_FEATURE_COLUMNS, *TARGET_FEATURE_COLUMNS]
                if col in row and pd.notna(row[col])
            }  # unknown-at-this-bin features are OMITTED, not written as None/NaN —
               # GML has no native null, so a stored None/NaN would round-trip as the
               # literal string "None", which a consumer would silently misparse as
               # a real value. Downstream code must treat a missing key as "unknown",
               # e.g. `data.get("sinr_dl_db")` returning None, not `data["sinr_dl_db"]`.
            snapshots[t].add_node(
                node_id, type="ue", lat=float(lat), lon=float(lon),
                vehicle_id=row.get("vehicle_id"), **features,
            )
            snapshots[t].add_edge(
                node_id, f"gnb_{gnb_id}",
                weight=features.get("sinr_dl_db", 0.0),
                distance_m=dist_m, in_range=int(in_range),
            )

    return bin_starts, snapshots


def stitch_part_and_transition(
    main_bin_starts: list[float], main_snapshots: dict[float, nx.Graph],
    trans_bin_starts: list[float], trans_snapshots: dict[float, nx.Graph],
    node_mapping: dict[int, int], main_part_label: str, trans_part_label: str,
) -> tuple[list[float], dict[float, nx.Graph]]:
    """Concatenates a main-part sequence with its following transition-window
    sequence into one continuous sequence, relabeling the transition part's UE
    node ids to the main part's ids wherever node_mapping provides the link
    (mapping is original_id [main-part-local id] -> sequential_id [transition-
    part-local id]). UEs in the transition part with no entry in node_mapping
    are new arrivals and keep their own (prefixed) identity.
    """
    reverse_map = {seq_id: orig_id for orig_id, seq_id in node_mapping.items()}
    relabeled_trans = {}
    for t, g in trans_snapshots.items():
        mapping = {}
        for node, data in g.nodes(data=True):
            if data.get("type") != "ue":
                continue
            trans_local_id = int(node.rsplit("_", 1)[-1])
            if trans_local_id in reverse_map:
                mapping[node] = f"ue_{main_part_label}_{reverse_map[trans_local_id]}"
        relabeled_trans[t] = nx.relabel_nodes(g, mapping, copy=True)

    combined_bins = list(main_bin_starts) + list(trans_bin_starts)
    combined_snapshots = {**main_snapshots, **relabeled_trans}
    return combined_bins, combined_snapshots


# =====================================================================
# Coverage / quality reporting — how much of each snapshot is carried-forward
# staleness vs. fresh events, so bin_size_s can be tuned with evidence.
# =====================================================================

def summarize_sequence(bin_starts: list[float], snapshots: dict[float, nx.Graph]) -> dict:
    ue_counts = [
        sum(1 for _, d in snapshots[t].nodes(data=True) if d.get("type") == "ue")
        for t in bin_starts
    ]
    in_range_frac = []
    known_sinr_frac = []
    for t in bin_starts:
        g = snapshots[t]
        ue_edges = [
            (u, v, d) for u, v, d in g.edges(data=True)
            if g.nodes[u].get("type") == "ue" or g.nodes[v].get("type") == "ue"
        ]
        if ue_edges:
            in_range_frac.append(sum(1 for *_, d in ue_edges if d.get("in_range")) / len(ue_edges))
            known_sinr_frac.append(
                sum(1 for *_, d in ue_edges if d.get("weight") not in (0.0, None)) / len(ue_edges)
            )
    return {
        "n_bins": len(bin_starts),
        "n_gnb": sum(1 for _, d in snapshots[bin_starts[0]].nodes(data=True) if d.get("type") == "gnb") if bin_starts else 0,
        "ue_count_per_bin_min": min(ue_counts) if ue_counts else 0,
        "ue_count_per_bin_max": max(ue_counts) if ue_counts else 0,
        "ue_count_per_bin_mean": sum(ue_counts) / len(ue_counts) if ue_counts else 0.0,
        "mean_in_range_fraction": sum(in_range_frac) / len(in_range_frac) if in_range_frac else None,
        "mean_known_sinr_fraction": sum(known_sinr_frac) / len(known_sinr_frac) if known_sinr_frac else None,
    }


# =====================================================================
# B5G — static, real topologies (no reconstruction needed)
# =====================================================================

def _b5g_data_root(data_root: str | None = None) -> str:
    if data_root is not None:
        return data_root
    full = os.path.join(_DATA_RAW, "b5g_slicing", "2.2.1-10kprocessed")
    sample = os.path.join(_DATA_RAW, "b5g_slicing_sample", "2.2.1-10kprocessed")
    return full if os.path.isdir(full) else sample


def load_b5g_topology(graph_id: int = 0, data_root: str | None = None) -> nx.MultiDiGraph:
    """Loads one B5G GML topology (real graph, no inference needed)."""
    data_root = _b5g_data_root(data_root)
    path = os.path.join(data_root, "graphs", f"graph_{graph_id}.txt")
    return nx.read_gml(path)


def load_b5g_topologies(
    max_graphs: int | None = None, data_root: str | None = None
) -> dict[int, nx.MultiDiGraph]:
    """Loads several B5G GML topologies (FR3 — generalize across unseen
    topologies). Returns {graph_id: graph}, sorted by graph_id, so the caller
    can hold some out entirely for the generalization test rather than
    splitting time windows of a single topology."""
    data_root = _b5g_data_root(data_root)
    graph_dir = os.path.join(data_root, "graphs")
    ids = sorted(
        int(re.search(r"graph_(\d+)\.txt$", p).group(1))
        for p in glob.glob(os.path.join(graph_dir, "graph_*.txt"))
    )
    if max_graphs:
        ids = ids[:max_graphs]
    return {gid: load_b5g_topology(gid, data_root) for gid in ids}


def split_b5g_graphs_for_generalization(
    graph_ids: list[int], holdout_frac: float = 0.2, seed: int = 0
) -> tuple[list[int], list[int]]:
    """Splits B5G topology ids into train/holdout SETS OF TOPOLOGIES (not time
    windows) — FR3 asks whether the model generalizes to graphs it never saw,
    which a temporal split within one topology cannot test."""
    import random
    ids = sorted(graph_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_holdout = max(1, round(len(ids) * holdout_frac)) if ids else 0
    holdout = sorted(ids[:n_holdout])
    train = sorted(ids[n_holdout:])
    return train, holdout


# =====================================================================
# Persistence
# =====================================================================

def save_sequence(
    bin_starts: list[float], snapshots: dict[float, nx.Graph], out_dir: str, manifest_extra: dict | None = None,
) -> None:
    """Writes one GML file per bin (`snapshot_<index>.gml`) plus a manifest.json
    recording bin_size, bin_starts (so index -> real timestamp is recoverable),
    and coverage stats from summarize_sequence."""
    os.makedirs(out_dir, exist_ok=True)
    for i, t in enumerate(bin_starts):
        nx.write_gml(snapshots[t], os.path.join(out_dir, f"snapshot_{i:05d}.gml"), stringizer=str)
    manifest = {
        "n_snapshots": len(bin_starts),
        "bin_starts_s": bin_starts,
        "coverage": summarize_sequence(bin_starts, snapshots),
        **(manifest_extra or {}),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


# =====================================================================
# CLI
# =====================================================================

def _build_arg_parser():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--part-dir", default=os.environ.get(
        "NEVERSNET5G_PART_DIR", os.path.join(_DATA_RAW, "neversnet5g", "data", "part1")))
    p.add_argument("--transition-dir", default=None,
                    help="Optional following transition part (e.g. part1_5) to stitch on.")
    p.add_argument("--node-mapping", default=None,
                    help="node_mapping_*.txt for the transition window, required if --transition-dir is set.")
    p.add_argument("--gnb-metadata", default=GNB_METADATA_CSV)
    p.add_argument("--bin-size-s", type=float, default=10.0)
    p.add_argument("--max-ues", type=int, default=None)
    p.add_argument("--out-dir", default=os.path.join(_DATA_PROCESSED, "neversnet5g_graphs", "part1"))
    p.add_argument("--b5g-max-graphs", type=int, default=5)
    return p


def main():
    args = _build_arg_parser().parse_args()

    gnbs = load_gnb_metadata(args.gnb_metadata)
    print(f"Loaded {len(gnbs)} gNB positions from {args.gnb_metadata}")

    if not os.path.isdir(args.part_dir):
        print(f"Part folder not found at {args.part_dir} — set --part-dir or "
              f"NEVERSNET5G_PART_DIR / unzip the sample archive.")
        return

    bin_starts, snapshots = build_temporal_graph_sequence(
        args.part_dir, gnbs, bin_size_s=args.bin_size_s, max_ues=args.max_ues,
    )
    part_label = None
    m = _UE_FILENAME_RE.search(os.path.basename(discover_ue_files(args.part_dir)[0][1]))
    part_label = m.group("part") if m else "part"
    print(f"Built {len(bin_starts)} snapshots for {part_label} "
          f"(bin_size_s={args.bin_size_s})")

    if args.transition_dir and args.node_mapping:
        t_bins, t_snaps = build_temporal_graph_sequence(
            args.transition_dir, gnbs, bin_size_s=args.bin_size_s, max_ues=args.max_ues,
        )
        trans_label_match = _UE_FILENAME_RE.search(
            os.path.basename(discover_ue_files(args.transition_dir)[0][1]))
        trans_label = trans_label_match.group("part") if trans_label_match else "transition"
        mapping = load_node_mapping(args.node_mapping)
        bin_starts, snapshots = stitch_part_and_transition(
            bin_starts, snapshots, t_bins, t_snaps, mapping, part_label, trans_label,
        )
        print(f"Stitched on {trans_label} using {args.node_mapping} "
              f"({len(mapping)} UEs mapped) -> {len(bin_starts)} total snapshots")

    stats = summarize_sequence(bin_starts, snapshots)
    print("Coverage summary:", json.dumps(stats, indent=2))

    save_sequence(bin_starts, snapshots, args.out_dir, manifest_extra={
        "part_dir": args.part_dir, "bin_size_s": args.bin_size_s,
        "gnb_metadata": args.gnb_metadata,
    })
    print(f"Wrote {len(bin_starts)} snapshot GMLs + manifest.json to {args.out_dir}")

    b5g_graphs = load_b5g_topologies(max_graphs=args.b5g_max_graphs)
    if b5g_graphs:
        train_ids, holdout_ids = split_b5g_graphs_for_generalization(list(b5g_graphs.keys()))
        sizes = {gid: (g.number_of_nodes(), g.number_of_edges()) for gid, g in b5g_graphs.items()}
        print(f"Loaded {len(b5g_graphs)} B5G topologies {sizes}")
        print(f"FR3 generalization split -> train topologies {train_ids}, holdout topologies {holdout_ids}")
    else:
        print("No B5G topology files found — check data/raw/b5g_slicing(_sample)/.../graphs/")


if __name__ == "__main__":
    main()
