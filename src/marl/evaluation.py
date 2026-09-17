"""Streaming evaluation and Review-1 evidence generation for Krish's MARL module."""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from .b5g_env import B5GSlicingEnv, SLICE_TYPES
from .maddpg import MADDPGTrainer


@dataclass
class EvaluationSummary:
    policy: str
    samples: int
    agents: int
    mean_reward: float
    qos_satisfaction_rate: float
    sla_violation_rate: float
    mean_utilization: float
    mean_reconfiguration_churn: float
    mean_latency_ms: float
    mean_packet_loss_rate: float
    approval_rate: float
    mean_inference_ms_per_sample: float


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def calibrate_delta(
    data_root: str | None = None,
    output_dir: str | Path = "reports/krish_marl",
) -> dict:
    """Report empirical delta distributions without claiming final SLA values."""
    root = Path(data_root) if data_root else Path(B5GSlicingEnv().data_root)
    values: dict[str, list[float]] = {slice_type: [] for slice_type in SLICE_TYPES}
    files = sorted((root / "slices").glob("slices_*.json"))
    for path in files:
        with path.open(encoding="utf-8") as handle:
            for slice_record in json.load(handle):
                slice_type = slice_record.get("type")
                if slice_type in values:
                    values[slice_type].append(float(slice_record["delta"]))
    distributions = {}
    for slice_type, slice_values in values.items():
        distributions[slice_type] = {
            "count": len(slice_values),
            "min": min(slice_values) if slice_values else None,
            "max": max(slice_values) if slice_values else None,
            "mean": sum(slice_values) / len(slice_values) if slice_values else None,
            "q05": _quantile(slice_values, 0.05),
            "q50": _quantile(slice_values, 0.50),
            "q95": _quantile(slice_values, 0.95),
            "within_unit_interval": all(0.0 <= value <= 1.0 for value in slice_values),
        }
    report = {
        "data_root": str(root),
        "sample_files": len(files),
        "slice_distributions": distributions,
        "interpretation": "Empirical distribution only; final SLA thresholds require the source paper delta definition.",
        "current_thresholds": {"eMBB": 0.5, "URLLC": 0.15, "mMTC": 0.8},
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "delta_calibration.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# B5G Delta Calibration Report", "",
        f"Scanned `{len(files)}` slice files under `{root}`.", "",
        "| Slice | Count | Min | Median | Max | Q05 | Q95 | Unit interval |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for slice_type, distribution in distributions.items():
        lines.append(
            f"| {slice_type} | {distribution['count']} | {distribution['min']!s} | "
            f"{distribution['q50']:.4f} | {distribution['max']!s} | "
            f"{distribution['q05']:.4f} | {distribution['q95']:.4f} | "
            f"{distribution['within_unit_interval']} |"
        )
    lines.extend(["", report["interpretation"]])
    (destination / "delta_calibration.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def equal_split(_observation) -> dict[str, float]:
    return {slice_type: 1.0 / len(SLICE_TYPES) for slice_type in SLICE_TYPES}


def heuristic_action(observation) -> dict[str, float]:
    action = equal_split(observation)
    focus = observation.slice_type
    action[focus] = min(1.0, action[focus] + 0.2)
    remainder = 1.0 - action[focus]
    for slice_type in SLICE_TYPES:
        if slice_type != focus:
            action[slice_type] = remainder / 2.0
    return action


def _evaluate_policy(
    env: B5GSlicingEnv,
    policy_name: str,
    action_factory: Callable,
    limit: int | None = None,
    start_index: int = 0,
    checkpoint_path: Path | None = None,
    checkpoint_interval: int = 250,
) -> EvaluationSummary:
    totals = {"reward": 0.0, "qos": 0, "sla": 0, "utilization": 0.0, "churn": 0.0, "latency": 0.0, "loss": 0.0, "approved": 0}
    sample_count = 0
    agent_count = 0
    inference_ms = 0.0
    env._cursor = max(0, start_index)
    while env._cursor < len(env._indices) and (limit is None or sample_count < limit):
        observations = env.reset()
        start = time.perf_counter()
        if hasattr(action_factory, "batch"):
            actions = action_factory.batch(observations)
        else:
            actions = {agent_id: action_factory(observation) for agent_id, observation in observations.items()}
        _, rewards, _, infos = env.step(actions)
        inference_ms += (time.perf_counter() - start) * 1000.0
        sample_count += 1
        agent_count += len(observations)
        for agent_id, reward in rewards.items():
            breakdown = infos[agent_id]["reward_breakdown"]
            metrics = infos[agent_id]["metrics"]
            totals["reward"] += reward
            totals["qos"] += int(bool(metrics["sla_ok"]))
            totals["sla"] += int(breakdown.sla_violation_penalty < 0.0)
            totals["utilization"] += metrics["allocation_share"]
            totals["churn"] += -breakdown.reconfiguration_penalty
            if metrics["predicted_latency_ms"] is not None:
                totals["latency"] += float(metrics["predicted_latency_ms"])
            if observations[agent_id].packet_loss_rate is not None:
                totals["loss"] += observations[agent_id].packet_loss_rate
            totals["approved"] += int(bool(metrics["sla_ok"]))
        if checkpoint_path and sample_count % checkpoint_interval == 0:
            checkpoint_path.write_text(
                json.dumps({
                    "policy": policy_name,
                    "next_index": env._cursor,
                    "samples": sample_count,
                    "agents": agent_count,
                    "totals": totals,
                }, indent=2),
                encoding="utf-8",
            )
    denominator = max(agent_count, 1)
    return EvaluationSummary(
        policy=policy_name,
        samples=sample_count,
        agents=agent_count,
        mean_reward=totals["reward"] / denominator,
        qos_satisfaction_rate=totals["qos"] / denominator,
        sla_violation_rate=totals["sla"] / denominator,
        mean_utilization=totals["utilization"] / denominator,
        mean_reconfiguration_churn=totals["churn"] / denominator,
        mean_latency_ms=totals["latency"] / denominator,
        mean_packet_loss_rate=totals["loss"] / denominator,
        approval_rate=totals["approved"] / denominator,
        mean_inference_ms_per_sample=inference_ms / max(sample_count, 1),
    )


def evaluate_b5g(
    data_root: str | None = None,
    limit: int | None = None,
    output_dir: str | Path = "reports/krish_marl",
    checkpoint: str | None = None,
    train_episodes: int = 0,
    start_index: int = 0,
    checkpoint_interval: int = 250,
) -> list[EvaluationSummary]:
    """Evaluate equal-split, heuristic, and trained policies with bounded memory."""
    import torch

    torch.set_num_threads(1)
    root_kwargs = {"data_root": data_root} if data_root else {}
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    policies: list[tuple[str, Callable]] = [
        ("equal_split", equal_split),
        ("heuristic", heuristic_action),
    ]
    trainer = MADDPGTrainer()
    if checkpoint:
        trainer.load(checkpoint)
    elif train_episodes:
        trainer.train(B5GSlicingEnv(**root_kwargs))
    policy_name = "trained_actor" if checkpoint or train_episodes else "untrained_actor_baseline"
    def trained_policy(observation):
        return trainer._action_dict(trainer.select_action(observation))

    trained_policy.batch = trainer.select_actions
    policies.append((policy_name, trained_policy))

    summaries = []
    for policy_name, action_factory in policies:
        environment = B5GSlicingEnv(load_topology=False, **root_kwargs)
        summaries.append(
            _evaluate_policy(
                environment,
                policy_name,
                action_factory,
                limit,
                start_index=start_index,
                checkpoint_path=destination / f"{policy_name}.progress.json",
                checkpoint_interval=checkpoint_interval,
            )
        )
    payload = [asdict(summary) for summary in summaries]
    (destination / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (destination / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=payload[0].keys())
        writer.writeheader()
        writer.writerows(payload)
    _write_markdown(payload, destination / "README.md")
    _write_plot(payload, destination / "policy_comparison.png")
    calibrate_delta(data_root=data_root, output_dir=destination)
    return summaries


def _write_plot(payload: list[dict], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    labels = [item["policy"] for item in payload]
    rewards = [item["mean_reward"] for item in payload]
    sla_rates = [item["sla_violation_rate"] for item in payload]
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(labels, rewards, color="#2f6f9f")
    axes[0].set_title("Mean reward")
    axes[1].bar(labels, sla_rates, color="#c45b4d")
    axes[1].set_title("SLA violation rate")
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _write_markdown(payload: list[dict], path: Path) -> None:
    lines = [
        "# Krish MARL Evaluation",
        "",
        "Streaming B5G replay comparison for equal-split, heuristic, and policy actions.",
        "The B5G release exposes the recorded `delta`; missing latency and packet-loss fields are reported as zero.",
        "",
        "| Policy | Samples | Mean reward | QoS satisfaction | SLA violation | Utilization | Churn | Inference ms/sample |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in payload:
        lines.append(
            f"| {item['policy']} | {item['samples']} | {item['mean_reward']:.4f} | "
            f"{item['qos_satisfaction_rate']:.4f} | {item['sla_violation_rate']:.4f} | "
            f"{item['mean_utilization']:.4f} | {item['mean_reconfiguration_churn']:.4f} | "
            f"{item['mean_inference_ms_per_sample']:.3f} |"
        )
    lines.extend([
        "",
        "Artifacts: `summary.csv`, `summary.json`, and `policy_comparison.png`.",
        "Threshold calibration remains provisional until the source paper's exact `delta` definition is verified.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Krish's MARL policies on B5G samples")
    parser.add_argument("--limit", type=int, default=None, help="Number of samples; omit for the full dataset")
    parser.add_argument("--train-episodes", type=int, default=0)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="reports/krish_marl")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--checkpoint-interval", type=int, default=250)
    args = parser.parse_args()
    results = evaluate_b5g(
        limit=args.limit,
        train_episodes=args.train_episodes,
        checkpoint=args.checkpoint,
        output_dir=args.output,
        start_index=args.start_index,
        checkpoint_interval=args.checkpoint_interval,
    )
    for result in results:
        print(asdict(result))
