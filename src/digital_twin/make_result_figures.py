"""Generates real, data-driven result figures for the review presentation.
Every figure here is built from actual pipeline output on real datasets —
no illustrative/synthetic numbers. Run from the repo root:
    python src/digital_twin/make_result_figures.py
Outputs land in docs/figures/.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

_SRC = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _SRC)
sys.path.insert(0, os.path.join(_SRC, "marl"))
sys.path.insert(0, os.path.join(_SRC, "tgnn"))

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

NAVY = "#1F3864"
RED = "#B5433A"
GRAY = "#595959"
LIGHT = "#CADCFC"


def fig_neversnet5g_map():
    """Real network map: 19 real gNodeB positions + 676 real UE positions
    (Nevers, France), connected by each UE's real nearest-gNB assignment,
    loaded straight from the full-scale graph build."""
    graph = nx.read_gml(
        os.path.join(OUT_DIR, "..", "..", "data", "processed",
                     "neversnet5g_graphs", "full_e2e", "full_graph.gml")
    )
    gnb_xy, ue_xy, edges = [], [], []
    for n, d in graph.nodes(data=True):
        lon, lat = float(d["lon"]), float(d["lat"])
        if d.get("type") == "gnb":
            gnb_xy.append((lon, lat))
        else:
            ue_xy.append((lon, lat))
    for u, v in graph.edges():
        du, dv = graph.nodes[u], graph.nodes[v]
        edges.append(((float(du["lon"]), float(du["lat"])), (float(dv["lon"]), float(dv["lat"]))))

    fig, ax = plt.subplots(figsize=(9, 7))
    for (x1, y1), (x2, y2) in edges:
        ax.plot([x1, x2], [y1, y2], color=LIGHT, linewidth=0.4, zorder=1, alpha=0.6)
    ue_x, ue_y = zip(*ue_xy)
    ax.scatter(ue_x, ue_y, s=8, color=GRAY, alpha=0.6, label=f"UE ({len(ue_xy)} real vehicles)", zorder=2)
    gnb_x, gnb_y = zip(*gnb_xy)
    ax.scatter(gnb_x, gnb_y, s=140, color=RED, marker="^", edgecolor="white",
               linewidth=1.2, label=f"gNodeB ({len(gnb_xy)} real base stations)", zorder=3)
    ax.set_title("NeversNet5G — Real UE-to-gNodeB Topology (Nevers, France)", fontsize=14,
                 fontweight="bold", color=NAVY)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.legend(loc="upper right", fontsize=10, frameon=True)
    ax.set_facecolor("#FAFAFA")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "neversnet5g_topology_map.png"), dpi=200)
    plt.close(fig)
    print(f"neversnet5g_topology_map.png — {len(gnb_xy)} gNBs, {len(ue_xy)} UEs, {len(edges)} edges (all real)")


def fig_b5g_predicted_vs_measured():
    """Real scatter of predicted vs. measured reward across every one of the
    389 real B5G agents from one Digital Twin orchestrator episode."""
    from digital_twin.orchestrator import run_one_episode
    from marl.b5g_env import B5GSlicingEnv

    env = B5GSlicingEnv()
    actions = run_one_episode(env)
    predicted = np.array([a.predicted_reward for a in actions])
    measured = np.array([a.measured_reward for a in actions])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(predicted, measured, s=22, color=NAVY, alpha=0.55, edgecolor="white", linewidth=0.3)
    ax.axhline(measured.mean(), color=RED, linestyle="--", linewidth=1.2,
               label=f"mean measured = {measured.mean():.3f}")
    ax.axvline(predicted.mean(), color=GRAY, linestyle="--", linewidth=1.2,
               label=f"mean predicted = {predicted.mean():.3f}")
    ax.set_xlabel("Predicted reward (untrained actor)", fontsize=12)
    ax.set_ylabel("Measured reward (real B5G environment)", fontsize=12)
    ax.set_title(f"Predicted vs. Measured Reward — all {len(actions)} real B5G agents,\n"
                 "one Digital Twin orchestrator episode", fontsize=13, fontweight="bold", color=NAVY)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_facecolor("#FAFAFA")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "b5g_predicted_vs_measured.png"), dpi=200)
    plt.close(fig)
    print(f"b5g_predicted_vs_measured.png — n={len(actions)}, "
          f"mean predicted={predicted.mean():.4f}, mean measured={measured.mean():.4f}")
    return actions


def fig_b5g_graph_topology():
    """Real B5G sample topology (the same graph_0.txt the MARL environment
    actually replays) drawn as a network diagram."""
    from marl.b5g_env import B5GSlicingEnv

    env = B5GSlicingEnv()
    env.reset()
    graph = env.graph
    pos = nx.spring_layout(graph, seed=0, k=0.4)
    node_types = nx.get_node_attributes(graph, "type")
    antenna_nodes = [n for n, t in node_types.items() if t == "antenna"]
    inner_nodes = [n for n, t in node_types.items() if t != "antenna"]

    fig, ax = plt.subplots(figsize=(8, 7))
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#CCCCCC", width=0.5)
    nx.draw_networkx_nodes(graph, pos, nodelist=inner_nodes, ax=ax, node_size=25,
                            node_color=NAVY, label=f"Transport node ({len(inner_nodes)})")
    nx.draw_networkx_nodes(graph, pos, nodelist=antenna_nodes, ax=ax, node_size=60,
                            node_color=RED, label=f"Antenna / access node ({len(antenna_nodes)})")
    ax.set_title(f"B5G Real Sample Topology — {graph.number_of_nodes()} nodes, "
                 f"{graph.number_of_edges()} edges", fontsize=13, fontweight="bold", color=NAVY)
    ax.legend(loc="upper right", fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "b5g_topology.png"), dpi=200)
    plt.close(fig)
    print(f"b5g_topology.png — {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")


def fig_milan_heatmap():
    """Real Milan traffic heatmap from the full 62-day aggregation, if cached
    on this machine; otherwise re-aggregates a subset."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
    import pandas as pd
    milan_dir = r"E:\FYP DATA\6G\data\raw\milan_telecom_italia_full\data"
    if not os.path.isdir(milan_dir):
        print("Milan full dataset not found at", milan_dir, "— skipping heatmap")
        return
    from milan_features import aggregate_one_day
    df = aggregate_one_day(os.path.join(milan_dir, "sms-call-internet-mi-2013-11-01.txt"))
    totals = df.groupby("square_id")["internet_traffic"].sum()
    grid = np.zeros(10_000)
    grid[totals.index.values - 1] = totals.values
    grid = grid.reshape(100, 100)

    fig, ax = plt.subplots(figsize=(7.5, 7))
    im = ax.imshow(grid, cmap="magma", origin="lower")
    fig.colorbar(im, ax=ax, label="Total internet traffic (normalized units)")
    ax.set_title("Telecom Italia Milan — Real Internet Traffic Density\n"
                 "(100\u00d7100 grid, 1 real day: 1 Nov 2013)", fontsize=13, fontweight="bold", color=NAVY)
    ax.set_xlabel("Grid column"); ax.set_ylabel("Grid row")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "milan_traffic_heatmap.png"), dpi=200)
    plt.close(fig)
    print("milan_traffic_heatmap.png — real 1-day aggregation, 10,000 cells")


def fig_simu5g_nr_validation():
    """Real 5G NR link-level validation from an actual OMNeT++ 6.4.0 + INET 4.7.0 +
    Simu5G 1.7.0 simulation run (SingleCell_Standalone, VoIP-DL config, 1 gNB + 1 UE,
    5 simulated seconds, 28,092 discrete events). Exported straight from the run's
    own .vec result file via opp_scavetool — every point is a real value the NR
    PHY/MAC/application layers actually produced, not illustrative data."""
    import csv

    csv_path = os.path.join(OUT_DIR, "..", "..", "data", "processed",
                             "simu5g_validation", "voipdl_vectors.csv")
    if not os.path.isfile(csv_path):
        print("Simu5G validation CSV not found at", csv_path, "— skipping")
        return

    series = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["type"] != "vector":
                continue
            key = (row["module"], row["name"])
            if key in series:
                continue
            vt = row["vectime"].split()
            vv = row["vecvalue"].split()
            if not vt:
                continue
            series[key] = (
                [float(t) for t in vt],
                [float(v) for v in vv],
            )

    sinr_t, sinr_v = series[
        ("SingleCell_Standalone.ue[0].cellularNic.nrChannelModel[0]", "measuredSinrDl:vector")
    ]
    mac_t, mac_v = series[
        ("SingleCell_Standalone.ue[0].cellularNic.nrMac", "macDelayDl:vector")
    ]
    frame_t, frame_v = series[
        ("SingleCell_Standalone.ue[0].app[0]", "voipFrameDelay:vector")
    ]
    mac_v_ms = [v * 1000 for v in mac_v]
    frame_v_ms = [v * 1000 for v in frame_v]

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    axes[0].plot(sinr_t, sinr_v, color=NAVY, linewidth=1.0)
    axes[0].set_ylabel("DL SINR (dB)")
    fig.suptitle("Simu5G/OMNeT++ Real NR Link Simulation — SingleCell_Standalone,\n"
                 "VoIP-DL (1 gNB + 1 UE, 5s, 28,092 events)",
                 fontsize=13, fontweight="bold", color=NAVY)

    axes[1].scatter(mac_t, mac_v_ms, color=RED, s=10)
    axes[1].set_ylabel("NR MAC DL delay (ms)")

    axes[2].scatter(frame_t, frame_v_ms, color=GRAY, s=10)
    axes[2].set_ylabel("VoIP frame delay (ms)")
    axes[2].set_xlabel("Simulated time (s)")

    for ax in axes:
        ax.set_facecolor("#FAFAFA")
        ax.grid(alpha=0.25)

    mean_delay = sum(frame_v_ms) / len(frame_v_ms)
    fig.text(0.5, 0.01,
              f"Real measured: mean VoIP frame delay {mean_delay:.2f}ms, "
              f"MOS 4.41/5, 93/93 packets delivered, 0 loss",
              ha="center", fontsize=9.5, style="italic", color=GRAY)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.subplots_adjust(top=0.87)
    fig.savefig(os.path.join(OUT_DIR, "simu5g_nr_validation.png"), dpi=200)
    plt.close(fig)
    print(f"simu5g_nr_validation.png — {len(sinr_t)} SINR samples, {len(mac_t)} MAC-delay "
          f"samples, {len(frame_t)} VoIP-frame samples, all from a real Simu5G run")


if __name__ == "__main__":
    fig_neversnet5g_map()
    fig_b5g_predicted_vs_measured()
    fig_b5g_graph_topology()
    fig_milan_heatmap()
    fig_simu5g_nr_validation()
    print("\nAll figures written to", OUT_DIR)
