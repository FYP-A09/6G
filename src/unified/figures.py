"""Figures for the unified-environment results. Every number is read from reports/unified/*.json.

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
REP, OUT = ROOT / "reports" / "unified", ROOT / "docs" / "figures"
BLUE, ORANGE, AQUA, GRAY, INK, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#898781", "#0b0b0b", "#7a5cc2"
J = lambda n: json.loads((REP / n).read_text(encoding="utf-8"))
sd = lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0


def _style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK)
    ax.set_ylabel(ylabel, color="#52514e")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e1e0d9", linewidth=1)
    ax.set_axisbelow(True)


def scaling():
    refs = J("marl_v2_scaling_refs.json")
    sizes = sorted(int(n) for n in refs)
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    series = {}
    for n in sizes:
        runs = J(f"marl_v2_scaling_{n}.json")["runs"]
        for cfg in {r["config"] for r in runs}:
            series.setdefault(cfg, {})[n] = [r["mean"] for r in runs if r["config"] == cfg]
    single = {n: max((series[c][n] for c in series if c.startswith("single")), key=np.mean) for n in sizes}
    for label, data, color in (("Shared-actor MARL", series["marl"], BLUE), ("Single centralised agent (best of 2 learning rates)", single, ORANGE)):
        ax.errorbar(range(len(sizes)), [np.mean(data[n]) for n in sizes], yerr=[sd(data[n]) for n in sizes], color=color, marker="o", markersize=6,
                    linewidth=2, capsize=4, label=label)
        for i, n in enumerate(sizes):
            ax.scatter([i] * len(data[n]), data[n], color=color, s=14, alpha=0.5, zorder=3)
    ax.plot(range(len(sizes)), [np.mean(refs[str(n)]["expert_forecast_tuned"]) for n in sizes], color=GRAY, linestyle="--", linewidth=1.6, label="Expert rule, tuned, with forecast")
    ax.plot(range(len(sizes)), [np.mean(refs[str(n)]["expert_forecast_default"]) for n in sizes], color=GRAY, linestyle=":", linewidth=1.6, label="Expert rule, untuned, with forecast")
    ax.plot(range(len(sizes)), [np.mean(refs[str(n)]["equal_split"]) for n in sizes], color="#bdbbb2", linestyle="-.", linewidth=1.4, label="Equal split")
    ax.set_xticks(range(len(sizes))); ax.set_xticklabels([str(s) for s in sizes]); ax.set_xlabel("Number of cells (agents)", color="#52514e")
    _style(ax, "Held-out week reward against network size", "Mean daily reward, 16-22 Dec")
    ax.set_ylim(0.2, 1.0); ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.tight_layout(); fig.savefig(OUT / "unified_marl_vs_single.png", dpi=200); plt.close(fig)


def prediction():
    w = J("forecast_value_v2.json")["heldout_week"]["results"]
    sizes = sorted(int(n) for n in w)
    keys = (("expert_current", "Current demand", GRAY), ("expert_forecast_v2", "TGNN forecast", BLUE), ("expert_oracle", "Perfect forecast", AQUA))
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    xs = np.arange(len(sizes))
    for x, n in zip(xs, sizes):
        ax.plot([x, x], [w[str(n)]["expert_current"]["mean"], w[str(n)]["expert_oracle"]["mean"]], color="#e1e0d9", linewidth=6, solid_capstyle="round", zorder=1)
    for k, label, color in keys:
        m = [w[str(n)][k]["mean"] for n in sizes]
        ax.errorbar(xs, m, yerr=[w[str(n)][k]["ci95"] for n in sizes], fmt="o", color=color, markersize=8, markeredgecolor="white", markeredgewidth=1.5, capsize=3, zorder=3, label=label)
        for x, v in zip(xs, m):
            ax.text(x + 0.1, v, f"{v:.3f}", va="center", fontsize=8, color="#52514e")
    ax.set_xticks(xs); ax.set_xticklabels([f"{n} cells" for n in sizes]); ax.set_xlim(-0.4, len(sizes) - 0.4)
    _style(ax, "Same expert rules, different knowledge of the next 10 minutes", "Mean daily reward, 16-22 Dec (95% CI, 7 days)")
    ax.set_ylim(0.7, 1.0); ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper right")
    fig.tight_layout(); fig.savefig(OUT / "unified_prediction_value.png", dpi=200); plt.close(fig)


def forecast_for_marl():
    fc, nofc = J("marl_v2_shift_ddpg_fc.json")["runs"], J("marl_v2_shift_ddpg_nofc.json")["runs"]
    ex = J("marl_v2_shift_experts.json")["results"]["baseline"]
    rows = [("MARL\nno forecast", [np.mean(r["baseline"]) for r in nofc], GRAY), ("MARL\nTGNN forecast", [np.mean(r["baseline"]) for r in fc], BLUE)]
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for j, (lab, v, color) in enumerate(rows):
        ax.plot([j, j], [np.mean(v) - sd(v), np.mean(v) + sd(v)], color=color, linewidth=2, zorder=2)
        ax.scatter([j - 0.12] * len(v), v, color=color, s=18, alpha=0.6, zorder=2)
        ax.scatter([j], [np.mean(v)], color=color, s=80, edgecolor="white", linewidth=2, zorder=3)
        ax.text(j + 0.1, np.mean(v), f"{np.mean(v):.3f} ± {sd(v):.3f}", va="center", fontsize=9, color="#52514e")
    for lab, key, ls in (("Tuned expert, with forecast", "expert_forecast_tuned_on_baseline", "--"), ("Tuned expert, current demand", "expert_current_tuned_on_baseline", ":")):
        ax.axhline(np.mean(ex[key]), color=AQUA, linestyle=ls, linewidth=1.5, label=lab)
    ax.set_xticks(range(2)); ax.set_xticklabels([r[0] for r in rows]); ax.set_xlim(-0.5, 1.7)
    _style(ax, "MARL with and without the forecast (25 cells, 3 seeds)", "Mean daily reward, 16-22 Dec")
    ax.set_ylim(0.7, 0.95); ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "unified_forecast_ablation.png", dpi=200); plt.close(fig)


def shift():
    ex = J("marl_v2_shift_experts.json")["results"]
    fc, nofc = J("marl_v2_shift_ddpg_fc.json")["runs"], J("marl_v2_shift_ddpg_nofc.json")["runs"]
    scen = [("baseline", "Recorded traffic"), ("growth_6g", "6G growth mix"), ("massive_iot", "Massive IoT")]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    series = [("Equal split", lambda s: [np.mean(ex[s]["equal_split"])], GRAY, "s"), ("Tuned expert, forecast", lambda s: [np.mean(ex[s]["expert_forecast_tuned_on_baseline"])], AQUA, "D"),
              ("MARL, forecast", lambda s: [np.mean(r[s]) for r in fc], BLUE, "o"), ("MARL, no forecast", lambda s: [np.mean(r[s]) for r in nofc], ORANGE, "o")]
    for k, (label, f, color, mk) in enumerate(series):
        for i, (s, _) in enumerate(scen):
            v = f(s); x = i + (k - 1.5) * 0.15
            ax.errorbar([x], [np.mean(v)], yerr=[sd(v)] if len(v) > 1 else None, fmt=mk, color=color, markersize=8, markeredgecolor="white", capsize=3, label=label if i == 0 else None)
    ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in scen])
    _style(ax, "Trained and tuned on ordinary traffic, scored without retraining", "Mean daily reward, 16-22 Dec")
    ax.set_ylim(0.55, 0.95); ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper right")
    fig.tight_layout(); fig.savefig(OUT / "unified_marl_shift.png", dpi=200); plt.close(fig)


def ssl():
    masked, scarf = J("ssl_nidd.json")["label_budgets"], J("ssl_nidd_scarf.json")["label_budgets"]
    budgets = ["50", "200", "1000", "5000", "20000", "all"]
    x = [50, 200, 1000, 5000, 20000, 60000]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for label, src, key, color, ls in (("LightGBM, raw features", scarf, "raw_lightgbm", INK, "-"), ("MLP, raw features", scarf, "raw_mlp", GRAY, "-"),
                                       ("Logistic regression, raw features", scarf, "raw_logreg", GRAY, "--"), ("Fine-tuned encoder, masked pretraining", masked, "ssl_finetune", ORANGE, "-"),
                                       ("Fine-tuned encoder, contrastive pretraining", scarf, "ssl_finetune", BLUE, "-"), ("Raw + contrastive embedding, logistic regression", scarf, "raw_plus_ssl_logreg", VIOLET, "--")):
        ax.plot(x, [src[b][key]["macro_f1_mean"] for b in budgets], color=color, linestyle=ls, marker="o", markersize=4.5, linewidth=1.8, label=label)
    ax.set_xscale("log"); ax.minorticks_off(); ax.set_xticks(x); ax.set_xticklabels(["50", "200", "1k", "5k", "20k", "all"])
    ax.set_xlabel("Labelled flows used for the classifier", color="#52514e")
    _style(ax, "5G-NIDD attack classification: does pretraining help?", "Macro-F1 on 81,731 unseen flows")
    ax.set_ylim(0.5, 1.0); ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "unified_ssl_label_efficiency.png", dpi=200); plt.close(fig)


def city_forecaster():
    v2 = J("forecast_city_v2.json")["regions"]
    regions = [("block_100", "100 block cells"), ("core_ring_300", "300 cells around"), ("outer_9600", "9,600 outer cells"), ("city_10000", "All 10,000 cells")]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9), sharey=True)
    for s, ax in enumerate(axes):
        for i, (k, lab) in enumerate(regions):
            a, b = v2[k]["tgnn_v1_5days"]["better_than_persistence"][s] * 100, v2[k]["tgnn_v2"]["better_than_persistence"][s] * 100
            ax.plot([i, i], [a, b], color="#e1e0d9", linewidth=5, solid_capstyle="round", zorder=1)
            ax.scatter([i], [a], color=GRAY, s=60, edgecolor="white", zorder=3, label="Trained on 5 days" if (i == 0 and s == 0) else None)
            ax.scatter([i], [b], color=BLUE, s=60, edgecolor="white", zorder=3, label="Trained on 39 days" if (i == 0 and s == 0) else None)
        ax.axhline(0, color=INK, linewidth=1)
        ax.set_xticks(range(4)); ax.set_xticklabels([l.replace(" ", "\n", 1) for _, l in regions], fontsize=8)
        _style(ax, ["eMBB", "URLLC", "mMTC"][s], "Better than persistence (%)" if s == 0 else "")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("City-wide forecaster on the held-out week (16-22 Dec): more training, same model", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "unified_forecast_city.png", dpi=200); plt.close(fig)


def marl_best():
    ex = J("marl_v2_tuned_expert.json"); refs = J("marl_v2_refs.json")["policies"]
    best, plus = J("marl_v2_best.json"), J("marl_v2_plus.json")["variants"]
    rows = [("Equal split", [np.mean(refs["equal_split"]["per_day"])], GRAY, "s"),
            ("Expert rules, untuned, forecast", [np.mean(refs["expert_forecast"]["per_day"])], GRAY, "D"),
            ("Expert rules, tuned, current demand", [ex["current"]["test_mean"]], AQUA, "D"),
            ("Expert rules, tuned, forecast", [ex["forecast"]["test_mean"]], AQUA, "D"),
            ("MARL, 2000 episodes, final checkpoint", [np.mean(r["per_day"]) for r in J("marl_v2_ddpg_fc.json")["runs"]], ORANGE, "o"),
            ("MARL, 4000 episodes, validation-selected", [v["test_mean"] for v in best["seeds"].values()], BLUE, "o"),
            ("MARL, ensemble of those 3", [best["ensemble"]["test_mean"]], BLUE, "*"),
            ("MARL + neighbour communication (1 seed)", [plus["comm"]["0"]["test_mean"]], VIOLET, "o"),
            ("MARL + expert residual (1 seed)", [plus["res"]["0"]["test_mean"]], VIOLET, "o"),
            ("Expert rules, true next demand", [np.mean(refs["expert_oracle"]["per_day"])], INK, "D")]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, (label, v, color, mk) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.scatter(v, [y] * len(v), color=color, s=70 if mk != "*" else 150, marker=mk, edgecolor="white", linewidth=1.2, zorder=3, alpha=0.9)
        ax.text(max(v) + 0.006, y, f"{np.mean(v):.3f}", va="center", fontsize=8.5, color="#52514e")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=9)
    ax.axvline(ex["forecast"]["test_mean"], color=AQUA, linestyle="--", linewidth=1, zorder=1)
    ax.set_xlim(0.72, 0.97); ax.grid(axis="x", color="#e1e0d9"); ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Mean daily reward on the held-out week, 25 cells (dots are seeds)", color="#52514e")
    ax.set_title("Long-trained MARL passes the tuned expert", loc="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout(); fig.savefig(OUT / "unified_marl_best.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (prediction, forecast_for_marl, shift, ssl, city_forecaster, marl_best):
        f()
    if all((REP / f"marl_v2_scaling_{n}.json").exists() for n in (9, 25, 100)) and (REP / "marl_v2_scaling_refs.json").exists():
        scaling()
    print("figures written to", OUT)
