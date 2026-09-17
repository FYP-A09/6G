import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from digital_twin.orchestrator import run_one_episode
from marl.b5g_env import B5GSlicingEnv
from marl.maddpg import MADDPGTrainer, TrainingConfig
from marl.telemetry import TelemetryRewardDataset
from marl.evaluation import evaluate_b5g


@pytest.fixture
def env() -> B5GSlicingEnv:
    return B5GSlicingEnv()


def test_environment_exposes_full_reward_and_state(env: B5GSlicingEnv) -> None:
    observations = env.reset(predicted_demand={"node_29_slice_0": 25_000_000.0})
    agent_id = "node_29_slice_0"
    assert observations[agent_id].predicted_demand_bps == 25_000_000.0
    assert len(observations[agent_id].to_vector()) == 13

    _, rewards, dones, infos = env.step({agent_id: {"eMBB": 1.0}})
    breakdown = infos[agent_id]["reward_breakdown"]
    assert dones[agent_id] is True
    assert rewards[agent_id] == pytest.approx(breakdown.total)
    assert breakdown.reconfiguration_penalty < 0.0


def test_environment_rejects_invalid_allocations(env: B5GSlicingEnv) -> None:
    env.reset()
    with pytest.raises(ValueError, match="sum"):
        env.step({"node_29_slice_0": {"eMBB": 0.8, "URLLC": 0.5}})


def test_reward_uses_active_slice_allocation(env: B5GSlicingEnv) -> None:
    observations = env.reset()
    agent_id = "node_29_slice_0"
    _, low_rewards, _, _ = env.step({agent_id: {"eMBB": 0.1}})
    observations = env.reset()
    _, high_rewards, _, _ = env.step({agent_id: {"eMBB": 0.9}})
    assert high_rewards[agent_id] > low_rewards[agent_id]


def test_missing_qos_metrics_are_unavailable(env: B5GSlicingEnv) -> None:
    env.reset()
    agent_id = "node_29_slice_0"
    _, _, _, infos = env.step({agent_id: {"eMBB": 1.0}})
    metrics = infos[agent_id]["metrics"]
    assert metrics["latency_ok"] is None
    assert metrics["jitter_ok"] is None
    assert metrics["reliability_ok"] is None
    assert metrics["sla_ok"] is None
    assert metrics["sla_available"] is False


def test_trainer_runs_one_offline_episode(env: B5GSlicingEnv) -> None:
    trainer = MADDPGTrainer(TrainingConfig(episodes=1, seed=2))
    history = trainer.train(env)
    report = trainer.evaluate(env)
    assert len(history) == 1
    assert len(trainer.replay) == 389
    assert report["inference_ms_per_agent"] >= 0.0


def test_twin_receives_policy_candidates(env: B5GSlicingEnv) -> None:
    actions = run_one_episode(env, MADDPGTrainer(TrainingConfig(seed=2)))
    assert len(actions) == 389
    assert all(sum(action.slice_allocations.values()) <= 1.0 + 1e-6 for action in actions)
    assert all(action.predicted_reward != 0.0 for action in actions)


def test_telemetry_reward_adapter() -> None:
    path = ROOT / "data" / "raw" / "6g_ran_telemetry_fl" / "6g_fl_telemetry_200_clients.csv"
    summary = TelemetryRewardDataset(path).summary(limit=5)
    assert summary["samples"] == 5.0
    assert 0.0 <= summary["next_sla_rate"] <= 1.0


def test_telemetry_multistep_training() -> None:
    path = ROOT / "data" / "raw" / "6g_ran_telemetry_fl" / "6g_fl_telemetry_200_clients.csv"
    from marl.telemetry import TelemetryWindowEnv

    telemetry_env = TelemetryWindowEnv(path, max_windows=2)
    trainer = MADDPGTrainer(TrainingConfig(seed=4))
    history = trainer.train_multistep(telemetry_env, episodes=1)
    assert len(history) == 1
    assert len(trainer.replay) > 0


def test_b5g_evaluation_writes_evidence(tmp_path: Path) -> None:
    results = evaluate_b5g(limit=1, output_dir=tmp_path)
    assert {result.policy for result in results} == {"equal_split", "heuristic", "untrained_actor_baseline"}
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "summary.csv").exists()
