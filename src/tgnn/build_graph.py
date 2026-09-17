"""
Graph construction for the TGNN stage (Thrishala S N's task — see
docs/team/thrishala_sn/tasks.md, design.md, and docs/architecture/graph_schema_draft.md).

Two topology sources, per the draft schema:
1. NeversNet5G: UE/gNB graph, built here. gNB nodes + positions come straight from
   the metadata CSV; UE-gNB "serving cell" edges are NOT directly labeled in the
   per-UE CSVs (confirmed absent — the release's own DATA_DESCRIPTION.md says so),
   so this module reconstructs them via a nearest-gNB-by-distance heuristic. This
   was flagged as an open question for Thrishala to validate on the sample before
   trusting it at scale — this is a first attempt at that validation, not the
   final word.
2. B5G: already a real graph (GML files) — loaded directly, no inference needed.

Requires: networkx (available), pandas (available). No torch needed for this file.
"""

from __future__ import annotations

import csv
import glob
import math
import os

import networkx as nx
import pandas as pd

_DATA_RAW = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
GNB_METADATA_CSV = os.path.join(
    _DATA_RAW, "neversnet5g_metadata", "gnodeb_opencellid_selected_19.csv"
)


def load_b5g_topology(graph_id: int = 0, data_root: str | None = None) -> nx.MultiDiGraph:
    """Loads one B5G GML topology (real graph, no inference needed)."""
    if data_root is None:
        full = os.path.join(_DATA_RAW, "b5g_slicing", "2.2.1-10kprocessed")
        sample = os.path.join(_DATA_RAW, "b5g_slicing_sample", "2.2.1-10kprocessed")
        data_root = full if os.path.isdir(full) else sample
    path = os.path.join(data_root, "graphs", f"graph_{graph_id}.txt")
    return nx.read_gml(path)


def load_gnb_positions(metadata_csv: str = GNB_METADATA_CSV) -> dict[int, tuple[float, float]]:
    """Returns {gnb_index: (lat, lon)} from the NeversNet5G release's gNB metadata."""
    positions = {}
    with open(metadata_csv) as f:
        for row in csv.DictReader(f):
            positions[int(row["gnb_index"])] = (float(row["lat"]), float(row["lon"]))
    return positions


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters — good enough at city scale (Nevers, France)."""
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def nearest_gnb(lat: float, lon: float, gnb_positions: dict[int, tuple[float, float]]) -> int:
    """Nearest-gNB-by-distance heuristic — the reconstruction method flagged as an
    open question in graph_schema_draft.md. Ignores handover hysteresis, building
    penetration loss, and actual SINR — a starting approximation, not ground truth."""
    return min(
        gnb_positions,
        key=lambda gnb_id: _haversine_m(lat, lon, *gnb_positions[gnb_id]),
    )


# Only the columns actually needed for graph construction — the full schema has
# 23 columns; reading just these 4 is what makes the full 27GB/716-file run
# tractable (see build_full_ue_gnb_graph).
_GRAPH_COLUMNS = ["vehicle_id", "latitude", "longitude", "sinr_dl_db"]


def _add_ue_from_file(graph: nx.Graph, ue_file: str, gnb_positions: dict) -> bool:
    """Reads one per-UE CSV and adds its node + nearest-gNB edge to `graph` in
    place. Returns False (and adds nothing) if the file has no valid position
    rows — some NeversNet5G files are position-sparse per DATA_DESCRIPTION.md."""
    try:
        df = pd.read_csv(ue_file, usecols=_GRAPH_COLUMNS, low_memory=False)
    except ValueError:
        # A handful of files omit a subset of the 23-column canonical schema
        # (documented in DATA_DESCRIPTION.md §4) — fall back to whatever's present.
        df = pd.read_csv(ue_file, low_memory=False)
        for col in _GRAPH_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA
    except Exception as e:
        # A few of the larger files (e.g. part2's ~127MB UE traces) have hit the
        # C parser's tokenizer with an "out of memory" ParserError on this
        # machine — the underlying file isn't corrupt, the default parser buffer
        # just chokes on it. Retry once with the slower pure-Python engine before
        # giving up on this file (skip rather than crash the full-scale run).
        try:
            df = pd.read_csv(ue_file, usecols=_GRAPH_COLUMNS, engine="python")
        except Exception:
            print(f"  [skip] {os.path.basename(ue_file)}: unreadable even with the "
                  f"python engine ({e})")
            return False

    valid_pos = df.dropna(subset=["latitude", "longitude"])
    if valid_pos.empty:
        return False

    last = valid_pos.iloc[-1]
    vehicle_id = last.get("vehicle_id", os.path.basename(ue_file))
    lat, lon = float(last["latitude"]), float(last["longitude"])
    serving_gnb = nearest_gnb(lat, lon, gnb_positions)
    mean_sinr = df["sinr_dl_db"].mean(skipna=True) if "sinr_dl_db" in df else pd.NA

    node_id = f"ue_{vehicle_id}_{os.path.basename(ue_file)}"  # file-qualified: part
    # folders reuse small UE-id ranges (see DATA_DESCRIPTION.md §2), so the raw
    # vehicle_id alone is not unique across the full dataset without the
    # node_mapping_*.txt reconciliation files — qualifying by source file avoids
    # silently merging distinct vehicles until that reconciliation is done.
    graph.add_node(node_id, type="ue", lat=lat, lon=lon, vehicle_id=str(vehicle_id))
    graph.add_edge(
        node_id, f"gnb_{serving_gnb}",
        weight=float(mean_sinr) if pd.notna(mean_sinr) else 0.0,
    )
    return True


def build_ue_gnb_graph(
    part_dir: str, gnb_positions: dict[int, tuple[float, float]], max_ues: int | None = None
) -> nx.Graph:
    """
    Builds a UE-gNB graph from one NeversNet5G part-folder's per-UE CSVs.

    Node types: 'gnb' (static) and 'ue' (one snapshot position per UE — its last
    valid (lat, lon) reading in the file, not a full trajectory; extend to a
    time-indexed multigraph once the TGNN's temporal windowing is decided).
    Edges: UE -> nearest gNB (see nearest_gnb's caveats above), weighted by
    the UE's mean sinr_dl_db as a first proxy for edge "quality."
    """
    graph = nx.Graph()
    for gnb_id, (lat, lon) in gnb_positions.items():
        graph.add_node(f"gnb_{gnb_id}", type="gnb", lat=lat, lon=lon)

    ue_files = sorted(glob.glob(os.path.join(part_dir, "*_ue_*_metrics.csv")))
    if max_ues:
        ue_files = ue_files[:max_ues]

    for ue_file in ue_files:
        _add_ue_from_file(graph, ue_file, gnb_positions)

    return graph


def build_full_ue_gnb_graph(
    data_root: str, gnb_positions: dict[int, tuple[float, float]], progress_every: int = 100
) -> nx.Graph:
    """
    Same construction as build_ue_gnb_graph, scaled to every part-folder in the
    full 716-file / 27GB release (Keerthivasan's task: apply Thrishala's
    sample-validated schema to the full dataset — see
    docs/team/keerthivasan/tasks.md). Reads only the 4 needed columns per file to
    keep this tractable at full scale.
    """
    graph = nx.Graph()
    for gnb_id, (lat, lon) in gnb_positions.items():
        graph.add_node(f"gnb_{gnb_id}", type="gnb", lat=lat, lon=lon)

    ue_files = sorted(glob.glob(os.path.join(data_root, "*", "*_ue_*_metrics.csv")))
    added, skipped_empty = 0, 0
    for i, ue_file in enumerate(ue_files):
        if _add_ue_from_file(graph, ue_file, gnb_positions):
            added += 1
        else:
            skipped_empty += 1
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  ...{i + 1}/{len(ue_files)} files processed "
                  f"({added} UEs added, {skipped_empty} skipped - no valid position)")

    print(f"Full-scale build: {added} UE nodes from {len(ue_files)} files "
          f"({skipped_empty} skipped - no valid position rows)")
    return graph


if __name__ == "__main__":
    gnb_positions = load_gnb_positions()
    print(f"Loaded {len(gnb_positions)} gNB positions")

    # Point at the extracted sample locally; falls back to a clear message if absent.
    sample_part_dir = os.environ.get(
        "NEVERSNET5G_PART_DIR",
        os.path.join(_DATA_RAW, "neversnet5g", "data", "part1"),
    )
    if os.path.isdir(sample_part_dir):
        graph = build_ue_gnb_graph(sample_part_dir, gnb_positions, max_ues=10)
        ue_nodes = [n for n, d in graph.nodes(data=True) if d["type"] == "ue"]
        print(f"Built UE-gNB graph: {graph.number_of_nodes()} nodes "
              f"({len(ue_nodes)} UEs + {len(gnb_positions)} gNBs), "
              f"{graph.number_of_edges()} edges")
        for ue in ue_nodes[:3]:
            neighbors = list(graph.neighbors(ue))
            print(f"  {ue} -> {neighbors} (weight={graph[ue][neighbors[0]]['weight']:.1f} dB SINR)")
    else:
        print(f"NeversNet5G part folder not found at {sample_part_dir} — "
              f"set NEVERSNET5G_PART_DIR or unzip Thrishala_S_N_neversnet5g_sample.zip")

    b5g_graph = load_b5g_topology()
    print(f"Loaded B5G topology: {b5g_graph.number_of_nodes()} nodes, "
          f"{b5g_graph.number_of_edges()} edges")
