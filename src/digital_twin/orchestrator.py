"""
End-to-end Digital Twin orchestrator (Keerthivasan's task — the piece that ties
Sriranjana's SSL, Thrishala's TGNN, and Krish's MARL environment together through
the "what-if" evaluation gate before any action reaches the Application Layer).

This is the first real, runnable closed loop, using each teammate's actual code
(not a redesign of it) against real B5G data:

  1. Load a real B5G sample graph + slice flows          (src/marl/b5g_env.py)
  2. Stand in for Sriranjana's SSL embeddings             (placeholder — her real
     encoder isn't trained yet; clearly marked, per her own tasks.md)
  3. Run Thrishala's TGNN over the graph to predict per-node demand
                                                            (src/tgnn/model.py)
  4. Build a candidate MARL action informed by that prediction
                                                            (a simple heuristic
     stand-in for Krish's not-yet-trained MADDPG agent)
  5. Evaluate the candidate action against the Digital Twin's SLA gate and
     either approve or reject it — this file's own contribution.

Every dataclass field matches docs/architecture/interface_contracts.md exactly,
so swapping any placeholder for the real trained module later is a drop-in
replacement, not a rewrite of this file.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import networkx as nx
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "marl"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tgnn"))
from b5g_env import DemandPrediction, B5GSlicingEnv  # noqa: E402
from maddpg import MADDPGTrainer  # noqa: E402
from model import SSL_EMBEDDING_DIM, TGNNPredictor  # noqa: E402


@dataclass
class CandidateAction:
    """Matches docs/architecture/interface_contracts.md §3 exactly."""

    agent_id: str
    slice_allocations: dict[str, float]
    predicted_reward: float
    twin_evaluation: str = "pending"  # "pending" | "approved" | "rejected"
    rejection_reason: str | None = None
    measured_reward: float | None = None
    prediction_error: float | None = None
    twin_metrics: dict[str, float | bool] = field(default_factory=dict)


def make_placeholder_ssl_embeddings(graph: nx.Graph, timesteps: int = 3) -> torch.Tensor:
    """
    Stand-in for Sriranjana's SSL encoder (src/ssl/masked_reconstruction.py),
    which isn't trained yet. Returns [N, T, D] random embeddings — the SHAPE is
    the real contract (D=64 per interface_contracts.md §1); the VALUES are not
    meaningful until her encoder replaces this function.
    """
    n = graph.number_of_nodes()
    torch.manual_seed(0)
    return torch.randn(n, timesteps, SSL_EMBEDDING_DIM)


def graph_to_adjacency(graph: nx.Graph) -> tuple[torch.Tensor, list[str]]:
    """Dense adjacency matrix + the node-id ordering used to build it, so callers
    can map back from row index to the graph's original node ids."""
    node_order = list(graph.nodes())
    adj = nx.to_numpy_array(graph, nodelist=node_order)
    return torch.tensor(adj, dtype=torch.float32), node_order


def propose_action_from_prediction(
    slice_type: str, predicted_load: float
) -> dict[str, float]:
    """
    Heuristic stand-in for Krish's not-yet-trained MADDPG agent: allocate more
    share to the slice under load, proportionally, capped so the total stays
    <= 1.0. Replace with a trained policy's output once B5GSlicingEnv is wired
    into an actual training loop (see docs/team/krish_s/tasks.md).
    """
    base = {"eMBB": 0.34, "URLLC": 0.33, "mMTC": 0.33}
    boost = min(0.3, max(0.0, predicted_load) * 0.3)
    base[slice_type] = min(1.0, base[slice_type] + boost)
    remaining = 1.0 - base[slice_type]
    others = [s for s in base if s != slice_type]
    for s in others:
        base[s] = remaining / len(others)
    return base


def evaluate_in_twin(
    action: CandidateAction,
    recorded_delta: float,
    slice_type: str,
    measured_reward: float,
    twin_metrics: dict[str, float | bool | None],
) -> None:
    """
    The Digital Twin's "what-if" gate (FR4 in docs/team/keerthivasan/requirements.md):
    no candidate action reaches the Application Layer without passing this check
    first. Mutates `action` in place, setting twin_evaluation and, if rejected,
    rejection_reason — matching interface_contracts.md §3 exactly.
    """
    action.measured_reward = measured_reward
    action.prediction_error = measured_reward - action.predicted_reward
    action.twin_metrics = twin_metrics
    if twin_metrics["sla_ok"] is not True:
        action.twin_evaluation = "rejected"
        availability = "unavailable" if not twin_metrics["sla_available"] else "failed"
        action.rejection_reason = (
            f"{slice_type} candidate SLA evaluation {availability}; "
            "delta is a continuous QoS-deviation signal, not an SLA cutoff"
        )
    else:
        action.twin_evaluation = "approved"


def run_one_episode(
    env: B5GSlicingEnv,
    trainer: MADDPGTrainer | None = None,
    checkpoint: str | None = None,
) -> list[CandidateAction]:
    trainer = trainer or MADDPGTrainer()
    if checkpoint:
        trainer.load(checkpoint)
    obs = env.reset()
    adjacency, node_order = graph_to_adjacency(env.graph)
    embeddings = make_placeholder_ssl_embeddings(env.graph)

    tgnn = TGNNPredictor()
    with torch.no_grad():
        predictions = tgnn(embeddings, adjacency)  # [N, 3] -> throughput, latency, load

    node_index = {node_id: i for i, node_id in enumerate(node_order)}

    predicted_demand: dict[str, float] = {}
    for agent_id, agent_obs in obs.items():
        row = node_index.get(str(agent_obs.node_id))
        # TGNNPredictor.forward() already applies softplus to throughput/latency
        # and sigmoid to load (see model.py) — read its output directly instead
        # of re-activating an already-activated value (double-sigmoid squashes
        # predicted_load toward 0.5, corrupting the SLA gate downstream).
        predicted_throughput = float(predictions[row, 0].item()) if row is not None else 0.0
        predicted_latency = float(predictions[row, 1].item()) if row is not None else 0.0
        predicted_load = float(predictions[row, 2].item()) if row is not None else 0.0
        agent_obs.prediction = DemandPrediction(
            entity_id=agent_id,
            slice_type=agent_obs.slice_type,
            predicted_throughput_bps=predicted_throughput,
            predicted_latency_ms=predicted_latency,
            predicted_load=predicted_load,
            confidence=1.0,
            horizon_start=0.0,
            horizon_end=1.0,
        )

    policy_actions = trainer.select_actions(obs)
    _, rewards, _, infos = env.step(policy_actions)
    actions: list[CandidateAction] = []
    for agent_id, agent_obs in obs.items():
        slice_allocations = policy_actions[agent_id]
        action = CandidateAction(
            agent_id=agent_id,
            slice_allocations=slice_allocations,
            predicted_reward=trainer.predict_reward(agent_obs, slice_allocations),
        )
        evaluate_in_twin(
            action,
            agent_obs.recorded_delta,
            agent_obs.slice_type,
            rewards[agent_id],
            infos[agent_id]["metrics"],
        )
        actions.append(action)

    return actions


if __name__ == "__main__":
    env = B5GSlicingEnv()
    actions = run_one_episode(env)

    approved = sum(1 for a in actions if a.twin_evaluation == "approved")
    rejected = len(actions) - approved
    print(f"Episode: {len(actions)} candidate actions evaluated by the Digital Twin")
    print(f"  approved: {approved}, rejected: {rejected}")
    print("\nSample decisions:")
    for a in actions[:5]:
        print(f"  {a.agent_id}: {a.twin_evaluation}"
              + (f" ({a.rejection_reason})" if a.rejection_reason else ""))
