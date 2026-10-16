"""Figures for the unified-environment results (reads reports/unified/results.json).

Run: python src/unified/figures.py  ->  docs/figures/unified_*.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "figures"
BLUE, ORANGE, AQUA, GRAY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#898781", "#0b0b0b"


def _style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK)
    ax.set_ylabel(ylabel, color="#52514e")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e1e0d9", linewidth=1)
    ax.set_axisbelow(True)


def scaling(r: dict) -> None:
    sizes = sorted(int(n) for n in r["scaling"])
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for key, color, label in (("marl", BLUE, "MARL (shared actor, neighbourhood critic)"),
                              ("single_agent", ORANGE, "Single centralised agent (tuned)")):
        xs, mean, sd = [], [], []
        for n in sizes:
            runs = [x["mean_reward"] for x in r["scaling"][str(n)][key]]
            if runs:
                xs.append(n); mean.append(np.mean(runs)); sd.append(np.std(runs))
        ax.errorbar(xs, mean, yerr=sd, color=color, marker="o", markersize=6, linewidth=2, capsize=4, label=label)
    heur = [r["heuristics"][str(n)]["headroom_current"]["mean_reward"] for n in sizes]
    eq = [r["heuristics"][str(n)]["equal_split"]["mean_reward"] for n in sizes]
    ax.plot(sizes, heur, color=GRAY, linestyle="--", linewidth=1.5, label="Expert rules (current demand)")
    ax.plot(sizes, eq, color=GRAY, linestyle=":", linewidth=1.5, label="Equal split")
    ax.set_xscale("log"); ax.minorticks_off(); ax.set_xticks(sizes); ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("Number of cells (agents)", color="#52514e")
    _style(ax, "Held-out reward vs. network size (3 seeds, mean ± sd)", "Mean reward, test days")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout(); fig.savefig(OUT / "unified_marl_vs_single.png", dpi=200); plt.close(fig)


def prediction(r: dict) -> None:
    sizes = sorted(int(n) for n in r["heuristics"])
    keys = (("headroom_current", "Current demand", GRAY), ("headroom_tgnn", "TGNN forecast", BLUE),
            ("headroom_oracle", "Perfect forecast", AQUA))
    # Dot plot (not bars) so a zoomed y-axis does not exaggerate differences.
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    xs = np.arange(len(sizes))
    vals = {k: [r["heuristics"][str(n)][k]["mean_reward"] for n in sizes] for k, _, _ in keys}
    for x in xs:
        ax.plot([x, x], [vals["headroom_current"][x], vals["headroom_oracle"][x]], color="#e1e0d9", linewidth=6, solid_capstyle="round", zorder=1)
    for k, label, color in keys:
        ax.scatter(xs, vals[k], s=70, color=color, edgecolor="white", linewidth=2, zorder=3, label=label)
        for x, v in zip(xs, vals[k]):
            ax.text(x + 0.09, v, f"{v:.3f}", va="center", fontsize=8, color="#52514e")
    ax.set_xticks(xs); ax.set_xticklabels([f"{n} cells" for n in sizes]); ax.set_xlim(-0.4, len(sizes) - 0.4)
    _style(ax, "Same expert rules, different knowledge of the next 10 minutes", "Mean reward, test days")
    ax.set_ylim(0.6, 1.0)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper right")
    fig.tight_layout(); fig.savefig(OUT / "unified_prediction_value.png", dpi=200); plt.close(fig)


def ablation(r: dict) -> None:
    a = r.get("ablation_marl_n25", {})
    rows = [(k, lab) for k, lab in (("none", "No forecast"), ("tgnn", "TGNN forecast"), ("oracle", "Perfect forecast")) if a.get(k)]
    if not rows:
        return
    # Dot plot (seeds + mean ± sd) rather than bars, since the y-axis does not start at zero.
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for j, (k, lab) in enumerate(rows):
        v = [x["mean_reward"] for x in a[k]]
        color = (GRAY, BLUE, AQUA)[j]
        ax.plot([j, j], [np.mean(v) - np.std(v), np.mean(v) + np.std(v)], color=color, linewidth=2, zorder=2)
        ax.scatter([j - 0.12] * len(v), v, color=color, s=18, alpha=0.6, zorder=2, label="Seeds" if j == 0 else None)
        ax.scatter([j], [np.mean(v)], color=color, s=80, edgecolor="white", linewidth=2, zorder=3)
        ax.text(j + 0.1, np.mean(v), f"{np.mean(v):.3f} ± {np.std(v):.3f}", va="center", fontsize=9, color="#52514e")
    ax.set_xticks(range(len(rows))); ax.set_xticklabels([lab for _, lab in rows]); ax.set_xlim(-0.5, len(rows) - 0.3)
    _style(ax, "MARL with and without the forecast (25 cells, 3 seeds)", "Mean reward, test days")
    ax.set_ylim(0.6, 0.9)
    fig.tight_layout(); fig.savefig(OUT / "unified_forecast_ablation.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    res = json.loads((ROOT / "reports" / "unified" / "results.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    scaling(res); prediction(res); ablation(res)
    print("figures written to", OUT)
