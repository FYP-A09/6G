"""
PettingZoo-shaped environment wrapping the B5G Network Slicing Dataset for offline
MARL prototyping (Krish S's task — see docs/team/krish_s/tasks.md, design.md, and
b5g_dataset_notes.md).

The B5G dataset (data/raw/b5g_slicing/, or the small in-repo
data/raw/b5g_slicing_sample/) is a labeled offline dataset, not a live simulator:
each sample already has a recorded slice configuration and a per-slice `delta`
value. So this environment replays recorded network states rather than simulating
the live consequence of an action — an agent's action is scored against how well
it matches (or improves on) the recorded provisioning for that sample.

Note on datanetAPI.py: the dataset ships a `datanetAPI.py` loader, but it expects
an older tar.gz-bundled release layout. This dataset's actual release
("2.2.1-10kprocessed") is already unpacked into flat `graphs/graph_N.txt`,
`routings/routing_N.txt`, `slices/slices_N.json` files sharing a common index N —
so this module reads that layout directly (via networkx for the GML graphs)
instead of importing the mismatched API.

One episode = one sample index N (one graph + slice configuration). One agent per
network node that originates at least one flow in that sample.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
from dataclasses import dataclass, field
from typing import Mapping

import networkx as nx

_DATA_RAW = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
_B5G_FULL = os.path.join(_DATA_RAW, "b5g_slicing", "2.2.1-10kprocessed")
_B5G_ATTACHED = os.path.join(
    os.path.dirname(__file__), "..", "..", "docs", "team", "krish_s",
    "b5g_slicing", "2.2.1-10kprocessed",
)
_B5G_SAMPLE = os.path.join(_DATA_RAW, "b5g_slicing_sample", "2.2.1-10kprocessed")
_B5G_ROOT = next(
    root for root in (_B5G_FULL, _B5G_ATTACHED, _B5G_SAMPLE) if os.path.isdir(root)
)

SLICE_TYPES = ("eMBB", "URLLC", "mMTC")

QOS_TARGETS = {
    "eMBB": {"latency_ms": 50.0, "jitter_ms": 20.0, "throughput_bps": 10_000_000.0, "reliability": 0.99},
    "URLLC": {"latency_ms": 5.0, "jitter_ms": 1.0, "throughput_bps": 1_000_000.0, "reliability": 0.99999},
    "mMTC": {"latency_ms": 100.0, "jitter_ms": 50.0, "throughput_bps": 100_000.0, "reliability": 0.99},
}

# Reward weights from docs/team/krish_s/design.md — starting values, not tuned.
REWARD_WEIGHTS = dict(
    w1_qos=1.0,           # QoS satisfaction (from 1 - delta, see _compute_reward)
    w2_utilization=0.3,   # resource utilization (bandwidth allocated / requested)
    w3_latency=0.5,
    w4_packet_loss=0.5,
    w5_sla_violation=1.5, # Reserved for explicit metric-based SLA violations.
    w6_reconfiguration=0.25,
)

# Farreras et al. (2024) does not define authoritative per-slice SLA cutoffs for
# this field. Delta remains a continuous QoS-deviation signal only; SLA decisions
# must use explicit QoS metrics below when those metrics are available.
SLA_DELTA_THRESHOLD: dict[str, float] = {}


@dataclass
class DemandPrediction:
    """TGNN output aligned to one MARL agent and one future control window."""

    entity_id: str
    slice_type: str
    predicted_throughput_bps: float
    predicted_latency_ms: float
    predicted_load: float
    confidence: float = 1.0
    horizon_start: float | None = None
    horizon_end: float | None = None


@dataclass
class AgentObservation:
    """One agent's (= one flow-originating node's) observation for the episode."""

    node_id: int
    slice_type: str
    demand_bandwidth_bps: float
    recorded_delta: float
    prediction: DemandPrediction | None = None
    current_prb_utilization: float = 0.0
    latency_ms: float | None = None
    jitter_ms: float | None = None
    packet_loss_rate: float | None = None
    previous_allocation: dict[str, float] = field(default_factory=dict)

    def to_vector(self) -> list[float]:
        """Stable numeric state for Gym/PettingZoo policies."""
        predicted = self.prediction.predicted_throughput_bps / 1e9 if self.prediction else 0.0
        predicted_latency = self.prediction.predicted_latency_ms / 100.0 if self.prediction else 0.0
        predicted_load = self.prediction.predicted_load if self.prediction else 0.0
        confidence = self.prediction.confidence if self.prediction else 0.0
        return [
            float(self.current_prb_utilization),
            float(predicted),
            float(self.demand_bandwidth_bps) / 1e9,
            float(self.latency_ms or 0.0) / 100.0,
            float(self.jitter_ms or 0.0) / 100.0,
            float(self.packet_loss_rate or 0.0),
            float(self.recorded_delta),
            float(predicted_latency),
            float(predicted_load),
            float(confidence),
            *[float(self.previous_allocation.get(slice_type, 0.0)) for slice_type in SLICE_TYPES],
        ]

    @property
    def predicted_demand_bps(self) -> float | None:
        return self.prediction.predicted_throughput_bps if self.prediction else None

    @predicted_demand_bps.setter
    def predicted_demand_bps(self, value: float | None) -> None:
        if value is None:
            self.prediction = None
            return
        self.prediction = DemandPrediction(
            entity_id=str(self.node_id),
            slice_type=self.slice_type,
            predicted_throughput_bps=float(value),
            predicted_latency_ms=self.latency_ms,
            predicted_load=self.current_prb_utilization,
        )


@dataclass(frozen=True)
class RewardBreakdown:
    qos_satisfaction: float
    utilization: float
    latency_penalty: float
    packet_loss_penalty: float
    sla_violation_penalty: float
    reconfiguration_penalty: float

    @property
    def total(self) -> float:
        return sum((
            self.qos_satisfaction,
            self.utilization,
            self.latency_penalty,
            self.packet_loss_penalty,
            self.sla_violation_penalty,
            self.reconfiguration_penalty,
        ))


def _available_sample_indices(root: str) -> list[int]:
    graph_files = glob.glob(os.path.join(root, "graphs", "graph_*.txt"))
    indices = []
    for f in graph_files:
        m = re.search(r"graph_(\d+)\.txt$", f)
        if m:
            indices.append(int(m.group(1)))
    return sorted(indices)


def _load_routing_matrix(path: str) -> list[list[int]]:
    with open(path) as f:
        return [[int(x) for x in row if x != ""] for row in csv.reader(f)]


class B5GSlicingEnv:
    """
    Minimal PettingZoo-ParallelEnv-shaped wrapper (method names/shapes match its
    API so swapping in the real base class later, once pettingzoo is installed,
    is a small change rather than a rewrite).
    """

    def __init__(self, data_root: str = _B5G_ROOT, load_topology: bool = True):
        self.data_root = data_root
        self.load_topology = load_topology
        self._indices = _available_sample_indices(data_root)
        if not self._indices:
            raise RuntimeError(
                f"No graph_*.txt files found under {data_root} — "
                f"run: python src/data/download_datasets.py b5g"
            )
        self._cursor = 0
        self.graph: nx.MultiDiGraph | None = None
        self.routing: list[list[int]] | None = None
        self.slices: list[dict] | None = None
        self.agents: list[str] = []
        self._observations: dict[str, AgentObservation] = {}
        self._previous_allocations: dict[str, dict[str, float]] = {}

    def reset(
        self,
        predicted_demand: Mapping[str, DemandPrediction | float] | None = None,
        previous_allocations: Mapping[str, Mapping[str, float]] | None = None,
    ) -> dict[str, AgentObservation]:
        """Advance to the next sample index (= next episode) and return per-agent obs."""
        idx = self._indices[self._cursor % len(self._indices)]
        self._cursor += 1

        if self.load_topology:
            self.graph = nx.read_gml(os.path.join(self.data_root, "graphs", f"graph_{idx}.txt"))
            self.routing = _load_routing_matrix(
                os.path.join(self.data_root, "routings", f"routing_{idx}.txt")
            )
        with open(os.path.join(self.data_root, "slices", f"slices_{idx}.json")) as f:
            self.slices = json.load(f)

        obs: dict[str, AgentObservation] = {}
        predicted_demand = predicted_demand or {}
        previous_allocations = previous_allocations or {}
        for slc in self.slices:
            slice_type = slc["type"]
            delta = slc["delta"]
            for flow in slc["flows"]:
                agent_id = f"node_{flow['origin_node']}_slice_{slc['number']}"
                metrics = slc.get("metrics", {})
                obs[agent_id] = AgentObservation(
                    node_id=flow["origin_node"],
                    slice_type=slice_type,
                    demand_bandwidth_bps=flow["bandwidth"],
                    recorded_delta=delta,
                    prediction=self._coerce_prediction(
                        predicted_demand.get(agent_id), agent_id, slice_type
                    ),
                    current_prb_utilization=float(metrics.get("prb_utilization", 0.0)),
                    latency_ms=self._optional_metric(metrics, "latency_ms"),
                    jitter_ms=self._optional_metric(metrics, "jitter_ms"),
                    packet_loss_rate=self._optional_metric(metrics, "packet_loss_rate"),
                    previous_allocation=dict(previous_allocations.get(agent_id, {})),
                )
        self.agents = list(obs.keys())
        self._observations = obs
        self._previous_allocations = {
            agent_id: dict(agent_obs.previous_allocation)
            for agent_id, agent_obs in obs.items()
        }
        return obs

    @staticmethod
    def _optional_metric(metrics: Mapping[str, object], name: str) -> float | None:
        value = metrics.get(name)
        return None if value is None else float(value)

    @staticmethod
    def _coerce_prediction(
        prediction: DemandPrediction | float | None,
        agent_id: str,
        slice_type: str,
    ) -> DemandPrediction | None:
        if prediction is None:
            return None
        if isinstance(prediction, DemandPrediction):
            if prediction.slice_type != slice_type:
                raise ValueError(f"Prediction slice type does not match {agent_id}")
            return prediction
        return DemandPrediction(
            entity_id=agent_id,
            slice_type=slice_type,
            predicted_throughput_bps=float(prediction),
            predicted_latency_ms=0.0,
            predicted_load=0.0,
        )

    def step(
        self, actions: dict[str, dict[str, float]]
    ) -> tuple[dict, dict[str, float], dict[str, bool], dict]:
        """
        actions: {agent_id: {slice_type: allocation_share}} — see
        interface_contracts.md §3 for the candidate-action shape.
        """
        if self.slices is None:
            raise RuntimeError("reset() must be called before step()")
        unknown_agents = set(actions) - set(self.agents)
        if unknown_agents:
            raise ValueError(f"Actions supplied for unknown agents: {sorted(unknown_agents)}")

        rewards = {}
        infos: dict[str, dict] = {}
        for agent_id, action in actions.items():
            # agent_id was constructed as f"node_{id}_slice_{n}" in reset()
            slice_num = int(agent_id.rsplit("_", 1)[1])
            slc = next(s for s in self.slices if s["number"] == slice_num)
            self._validate_action(action)
            observation = self._observation_for(agent_id)
            metrics = self._action_metrics(action, observation)
            breakdown = self._compute_reward(action, slc, observation, sla_ok=metrics["sla_ok"])
            rewards[agent_id] = breakdown.total
            infos[agent_id] = {
                "reward_breakdown": breakdown,
                "metrics": metrics,
            }

        dones = {agent_id: True for agent_id in self.agents}  # one-shot per sample
        next_obs: dict = {}  # caller should call reset() for the next sample
        return next_obs, rewards, dones, infos

    def _observation_for(self, agent_id: str) -> AgentObservation:
        try:
            return self._observations[agent_id]
        except KeyError as exc:
            raise RuntimeError("reset() must be called before step()") from exc

    def _validate_action(self, action: Mapping[str, float]) -> None:
        unknown = set(action) - set(SLICE_TYPES)
        if unknown:
            raise ValueError(f"Unknown slice types in action: {sorted(unknown)}")
        if any(value < 0.0 or value > 1.0 for value in action.values()):
            raise ValueError("Allocation shares must be within [0, 1]")
        if sum(action.values()) > 1.0 + 1e-6:
            raise ValueError("Allocation shares must sum to <= 1.0")

    def _action_metrics(
        self, action: Mapping[str, float], observation: AgentObservation
    ) -> dict[str, float | bool | None]:
        target = QOS_TARGETS[observation.slice_type]
        prediction = observation.prediction
        predicted_latency = prediction.predicted_latency_ms if prediction else observation.latency_ms
        predicted_load = prediction.predicted_load if prediction else observation.current_prb_utilization
        predicted_throughput = prediction.predicted_throughput_bps if prediction else observation.demand_bandwidth_bps
        allocation = action.get(observation.slice_type, 0.0)
        effective_throughput = predicted_throughput * allocation
        jitter_ok = None if observation.jitter_ms is None else observation.jitter_ms <= target["jitter_ms"]
        latency_ok = None if predicted_latency is None else predicted_latency <= target["latency_ms"]
        throughput_ok = effective_throughput >= target["throughput_bps"]
        reliability_ok = (
            None
            if observation.packet_loss_rate is None
            else 1.0 - observation.packet_loss_rate >= target["reliability"]
        )
        checks = (latency_ok, jitter_ok, throughput_ok, reliability_ok)
        sla_ok = None if any(check is None for check in checks) else all(checks)
        return {
            "predicted_latency_ms": predicted_latency,
            "predicted_load": predicted_load,
            "effective_throughput_bps": effective_throughput,
            "latency_ok": latency_ok,
            "jitter_ok": jitter_ok,
            "throughput_ok": throughput_ok,
            "reliability_ok": reliability_ok,
            "sla_ok": sla_ok,
            "sla_available": sla_ok is not None,
            "allocation_share": allocation,
        }

    def _compute_reward(
        self,
        action: Mapping[str, float],
        slc: dict,
        observation: AgentObservation | None = None,
        sla_ok: bool | None = None,
    ) -> RewardBreakdown:
        """Compute the six-term reward without treating missing metrics as passing SLA.

        `sla_ok` comes from `_action_metrics` (QOS_TARGETS-based latency/jitter/
        throughput/reliability checks) — delta has no paper-defined SLA cutoff
        (see SLA_DELTA_THRESHOLD's docstring above) so it is not used for the SLA
        term. `sla_ok is None` means the metric was unavailable (missing jitter/
        packet-loss data, etc.) and is treated as neutral, not a violation — only
        a confirmed `sla_ok is False` incurs the penalty.
        """
        w = REWARD_WEIGHTS
        slice_type = slc["type"]
        delta = slc["delta"]
        observation = observation or AgentObservation(0, slice_type, 0.0, delta)

        qos_term = w["w1_qos"] * max(0.0, 1.0 - delta)
        allocation_share = action.get(slice_type, 0.0)
        utilization_term = w["w2_utilization"] * min(1.0, allocation_share)
        latency_value = observation.prediction.predicted_latency_ms if observation.prediction else observation.latency_ms
        latency_penalty = -w["w3_latency"] * min(1.0, latency_value / 100.0) if latency_value is not None else 0.0
        packet_loss_penalty = (
            -w["w4_packet_loss"] * min(1.0, observation.packet_loss_rate)
            if observation.packet_loss_rate is not None else 0.0
        )
        sla_term = -w["w5_sla_violation"] if sla_ok is False else 0.0
        previous = observation.previous_allocation
        churn = sum(abs(action.get(kind, 0.0) - previous.get(kind, 0.0)) for kind in SLICE_TYPES)
        reconfiguration_term = -w["w6_reconfiguration"] * min(1.0, churn)

        return RewardBreakdown(
            qos_satisfaction=qos_term,
            utilization=utilization_term,
            latency_penalty=latency_penalty,
            packet_loss_penalty=packet_loss_penalty,
            sla_violation_penalty=sla_term,
            reconfiguration_penalty=reconfiguration_term,
        )


if __name__ == "__main__":
    env = B5GSlicingEnv()
    print(f"Loaded {len(env._indices)} sample(s) from {env.data_root}")
    obs = env.reset()
    print(f"Episode has {len(obs)} agents, graph has {env.graph.number_of_nodes()} nodes, "
          f"{env.graph.number_of_edges()} edges")
    for agent_id, o in list(obs.items())[:3]:
        print(" ", agent_id, o)

    # Smoke-test step() with a trivial equal-split action for each agent.
    dummy_actions = {aid: {"eMBB": 0.34, "URLLC": 0.33, "mMTC": 0.33} for aid in obs}
    _, rewards, _, _ = env.step(dummy_actions)
    print("Sample rewards:", dict(list(rewards.items())[:3]))
