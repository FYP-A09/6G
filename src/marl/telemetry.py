"""Adapter for the 6G RAN telemetry reward proxy owned by Krish's module."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

try:
    from .b5g_env import AgentObservation, SLICE_TYPES
except ImportError:
    from b5g_env import AgentObservation, SLICE_TYPES


@dataclass(frozen=True)
class TelemetryRewardSample:
    client_id: str
    window_idx: int
    timestamp_s: float
    reward: float
    sla_ok_current: bool
    sla_ok_next: bool
    slice_imbalance: float
    latency_p95_ms: float
    regime: str


class TelemetryRewardDataset:
    """Stream reward-labelled client windows without requiring pandas."""

    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)

    def __iter__(self) -> Iterator[TelemetryRewardSample]:
        with self.csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                yield TelemetryRewardSample(
                    client_id=row["client_id"],
                    window_idx=int(row["window_idx"]),
                    timestamp_s=float(row["timestamp_s"]),
                    reward=float(row["reward"]),
                    sla_ok_current=bool(int(row["sla_ok_current"])),
                    sla_ok_next=bool(int(row["sla_ok_next"])),
                    slice_imbalance=float(row["slice_imbalance"]),
                    latency_p95_ms=float(row["lat_p95_ms"]),
                    regime=row["regime"],
                )

    def summary(self, limit: int | None = None) -> dict[str, float]:
        samples = []
        for index, sample in enumerate(self):
            if limit is not None and index >= limit:
                break
            samples.append(sample)
        if not samples:
            return {"samples": 0.0, "mean_reward": 0.0, "next_sla_rate": 0.0}
        return {
            "samples": float(len(samples)),
            "mean_reward": sum(sample.reward for sample in samples) / len(samples),
            "next_sla_rate": sum(sample.sla_ok_next for sample in samples) / len(samples),
            "mean_latency_p95_ms": sum(sample.latency_p95_ms for sample in samples) / len(samples),
        }

    def rows(self, limit: int | None = None) -> list[TelemetryRewardSample]:
        samples = []
        for index, sample in enumerate(self):
            if limit is not None and index >= limit:
                break
            samples.append(sample)
        return samples


class TelemetryWindowEnv:
    """Sequential one-agent-per-client environment for multi-step training."""

    def __init__(self, csv_path: str | Path, max_windows: int | None = None):
        samples = TelemetryRewardDataset(csv_path).rows()
        by_client: dict[str, list[TelemetryRewardSample]] = {}
        for sample in samples:
            by_client.setdefault(sample.client_id, []).append(sample)
        self.windows = {
            client_id: sorted(rows, key=lambda row: row.window_idx)
            for client_id, rows in by_client.items()
        }
        self.client_ids = sorted(self.windows)
        self.max_windows = max_windows
        self._window_index = 0
        self.agents: list[str] = []

    @staticmethod
    def _slice_type(sample: TelemetryRewardSample) -> str:
        # "impaired" clients fall back to URLLC — the strictest target — since a
        # struggling connection is exactly the case we don't want to under-serve.
        return {"heavy": "eMBB", "medium": "URLLC", "light": "mMTC"}.get(
            sample.regime, "URLLC"
        )

    def _observation(self, client_id: str, sample: TelemetryRewardSample) -> AgentObservation:
        slice_type = self._slice_type(sample)
        return AgentObservation(
            node_id=hash(client_id) % 1_000_000,
            slice_type=slice_type,
            demand_bandwidth_bps=1_000_000.0,
            recorded_delta=1.0 - sample.reward,
            current_prb_utilization=sample.slice_imbalance,
            latency_ms=sample.latency_p95_ms,
            packet_loss_rate=0.0 if sample.sla_ok_current else 0.05,
            previous_allocation={slice_type: 1.0 if sample.sla_ok_current else 0.0},
        )

    def reset(self) -> dict[str, AgentObservation]:
        self._window_index = 0
        self.agents = [f"client_{client_id}" for client_id in self.client_ids]
        return {
            f"client_{client_id}": self._observation(client_id, rows[0])
            for client_id, rows in self.windows.items()
            if rows
        }

    def step(self, actions: dict[str, dict[str, float]]) -> tuple[dict, dict, dict, dict]:
        rewards: dict[str, float] = {}
        next_obs: dict[str, AgentObservation] = {}
        infos: dict[str, dict] = {}
        next_index = self._window_index + 1
        for agent_id, action in actions.items():
            client_id = agent_id.removeprefix("client_")
            rows = self.windows[client_id]
            current = rows[min(self._window_index, len(rows) - 1)]
            allocation = action.get(self._slice_type(current), 0.0)
            rewards[agent_id] = current.reward - abs(allocation - (1.0 if current.sla_ok_next else 0.0)) * 0.1
            infos[agent_id] = {"telemetry_window": current.window_idx, "measured_reward": current.reward}
            if next_index < len(rows) and (self.max_windows is None or next_index < self.max_windows):
                next_obs[agent_id] = self._observation(client_id, rows[next_index])
        self._window_index = next_index
        done = {agent_id: agent_id not in next_obs for agent_id in self.agents}
        return next_obs, rewards, done, infos


if __name__ == "__main__":
    default_path = Path(__file__).parents[2] / "data" / "raw" / "6g_ran_telemetry_fl" / "6g_fl_telemetry_200_clients.csv"
    print(TelemetryRewardDataset(default_path).summary(limit=1000))
