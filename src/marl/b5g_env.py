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

import networkx as nx

_DATA_RAW = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
_B5G_FULL = os.path.join(_DATA_RAW, "b5g_slicing", "2.2.1-10kprocessed")
_B5G_SAMPLE = os.path.join(_DATA_RAW, "b5g_slicing_sample", "2.2.1-10kprocessed")
_B5G_ROOT = _B5G_FULL if os.path.isdir(_B5G_FULL) else _B5G_SAMPLE

SLICE_TYPES = ("eMBB", "URLLC", "mIoT")

# Reward weights from docs/team/krish_s/design.md — starting values, not tuned.
REWARD_WEIGHTS = dict(
    w1_qos=1.0,           # QoS satisfaction (from 1 - delta, see _compute_reward)
    w2_utilization=0.3,   # resource utilization (bandwidth allocated / requested)
    w5_sla_violation=1.5, # SLA violation penalty (delta above the per-slice-type threshold)
)

# Rough per-slice-type SLA thresholds on `delta` (treated as a normalized
# delay/deviation metric) — placeholder values, calibrate against the dataset
# paper (Farreras et al., 2024) once read in full.
SLA_DELTA_THRESHOLD = {"eMBB": 0.5, "URLLC": 0.15, "mIoT": 0.8}


@dataclass
class AgentObservation:
    """One agent's (= one flow-originating node's) observation for the episode."""

    node_id: int
    slice_type: str
    demand_bandwidth_bps: float
    recorded_delta: float
    predicted_demand_bps: float | None = None  # filled in by the TGNN, once wired up
    previous_allocation: dict[str, float] = field(default_factory=dict)


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

    def __init__(self, data_root: str = _B5G_ROOT):
        self.data_root = data_root
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

    def reset(self) -> dict[str, AgentObservation]:
        """Advance to the next sample index (= next episode) and return per-agent obs."""
        idx = self._indices[self._cursor % len(self._indices)]
        self._cursor += 1

        self.graph = nx.read_gml(os.path.join(self.data_root, "graphs", f"graph_{idx}.txt"))
        self.routing = _load_routing_matrix(
            os.path.join(self.data_root, "routings", f"routing_{idx}.txt")
        )
        with open(os.path.join(self.data_root, "slices", f"slices_{idx}.json")) as f:
            self.slices = json.load(f)

        obs: dict[str, AgentObservation] = {}
        for slc in self.slices:
            slice_type = slc["type"]
            delta = slc["delta"]
            for flow in slc["flows"]:
                agent_id = f"node_{flow['origin_node']}_slice_{slc['number']}"
                obs[agent_id] = AgentObservation(
                    node_id=flow["origin_node"],
                    slice_type=slice_type,
                    demand_bandwidth_bps=flow["bandwidth"],
                    recorded_delta=delta,
                )
        self.agents = list(obs.keys())
        return obs

    def step(
        self, actions: dict[str, dict[str, float]]
    ) -> tuple[dict, dict[str, float], dict[str, bool], dict]:
        """
        actions: {agent_id: {slice_type: allocation_share}} — see
        interface_contracts.md §3 for the candidate-action shape.
        """
        rewards = {}
        for agent_id, action in actions.items():
            # agent_id was constructed as f"node_{id}_slice_{n}" in reset()
            slice_num = int(agent_id.rsplit("_", 1)[1])
            slc = next(s for s in self.slices if s["number"] == slice_num)
            rewards[agent_id] = self._compute_reward(action, slc)

        dones = {agent_id: True for agent_id in self.agents}  # one-shot per sample
        infos: dict = {}
        next_obs: dict = {}  # caller should call reset() for the next sample
        return next_obs, rewards, dones, infos

    def _compute_reward(self, action: dict[str, float], slc: dict) -> float:
        """Reward formula from design.md, using `delta` as the recorded QoS-deviation
        signal in place of a directly-simulated latency/loss value."""
        w = REWARD_WEIGHTS
        slice_type = slc["type"]
        delta = slc["delta"]
        threshold = SLA_DELTA_THRESHOLD.get(slice_type, 0.5)

        qos_term = w["w1_qos"] * (1.0 - delta)
        utilization_term = w["w2_utilization"] * sum(action.values())
        sla_violation = delta > threshold
        sla_term = -w["w5_sla_violation"] * float(sla_violation)

        return qos_term + utilization_term + sla_term


if __name__ == "__main__":
    env = B5GSlicingEnv()
    print(f"Loaded {len(env._indices)} sample(s) from {env.data_root}")
    obs = env.reset()
    print(f"Episode has {len(obs)} agents, graph has {env.graph.number_of_nodes()} nodes, "
          f"{env.graph.number_of_edges()} edges")
    for agent_id, o in list(obs.items())[:3]:
        print(" ", agent_id, o)

    # Smoke-test step() with a trivial equal-split action for each agent.
    dummy_actions = {aid: {"eMBB": 0.34, "URLLC": 0.33, "mIoT": 0.33} for aid in obs}
    _, rewards, _, _ = env.step(dummy_actions)
    print("Sample rewards:", dict(list(rewards.items())[:3]))
