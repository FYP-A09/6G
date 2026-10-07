import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from digital_twin.orchestrator import run_one_episode
from digital_twin.telemetry_replay import NeversNetTelemetryReplay
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


def test_neversnet_telemetry_replay_streams_real_schema(tmp_path: Path) -> None:
    part = tmp_path / "part1"
    part.mkdir()
    (part / "part1_ue_0_metrics.csv").write_text(
        "time_s,sinr_dl_db,throughput_dl_bps\n0.0,4.0,\n1.0,5.0,2000\n",
        encoding="utf-8",
    )
    replay = NeversNetTelemetryReplay([part], chunksize=1)
    events = list(replay.iter_events())
    assert [event.node_id for event in events] == ["ue_part1_0", "ue_part1_0"]
    assert events[-1].metrics["throughput_dl_bps"] == 2000.0
    assert replay.latest_state()["ue_part1_0"].timestamp_s == 1.0


def test_duplicate_slice_numbers_across_types_stay_separate_agents(tmp_path) -> None:
    """B5G slice numbers restart per type; agents/rewards must not merge across types."""
    import json

    (tmp_path / "graphs").mkdir()
    (tmp_path / "slices").mkdir()
    (tmp_path / "graphs" / "graph_0.txt").write_text("")
    flow = {"origin_node": 7, "destination": 1, "bandwidth": 1_000_000}
    slices = [
        {"number": 0, "type": "eMBB", "delta": 0.2, "flows": [flow]},
        {"number": 0, "type": "URLLC", "delta": 0.9, "flows": [flow]},
    ]
    (tmp_path / "slices" / "slices_0.json").write_text(json.dumps(slices))

    env = B5GSlicingEnv(data_root=str(tmp_path), load_topology=False)
    observations = env.reset()
    assert len(observations) == 2
    assert {o.slice_type for o in observations.values()} == {"eMBB", "URLLC"}

    actions = {
        agent_id: {"eMBB": 1.0} if obs.slice_type == "eMBB" else {"URLLC": 1.0}
        for agent_id, obs in observations.items()
    }
    _, _, _, infos = env.step(actions)
    for agent_id, obs in observations.items():
        assert infos[agent_id]["metrics"]["allocation_share"] == 1.0
        expected_qos = 1.0 - obs.recorded_delta
        assert infos[agent_id]["reward_breakdown"].qos_satisfaction == pytest.approx(expected_qos)


def test_agents_with_reused_slice_numbers_are_scored_against_their_own_slice(tmp_path: Path) -> None:
    # The real B5G release restarts slice "number" per slice type, so two slices of
    # different types can share number 0 and an origin node. They must stay separate
    # agents and be scored against their own type/delta.
    import json

    (tmp_path / "graphs").mkdir()
    (tmp_path / "graphs" / "graph_0.txt").write_text("")
    (tmp_path / "slices").mkdir()
    flow = {"origin_node": 7, "destination": 1, "bandwidth": 1_000_000}
    slices = [
        {"number": 0, "type": "eMBB", "delta": 0.2, "flows": [dict(flow)]},
        {"number": 0, "type": "mMTC", "delta": 0.9, "flows": [dict(flow)]},
    ]
    (tmp_path / "slices" / "slices_0.json").write_text(json.dumps(slices))

    env = B5GSlicingEnv(data_root=str(tmp_path), load_topology=False)
    observations = env.reset()
    assert len(observations) == 2
    assert sorted(o.slice_type for o in observations.values()) == ["eMBB", "mMTC"]

    actions = {
        agent_id: {"eMBB": 0.0, "URLLC": 0.0, "mMTC": 1.0}
        for agent_id in observations
    }
    _, _, _, infos = env.step(actions)
    for agent_id, observation in observations.items():
        expected_share = 1.0 if observation.slice_type == "mMTC" else 0.0
        assert infos[agent_id]["metrics"]["allocation_share"] == expected_share
        expected_qos = 1.0 - observation.recorded_delta
        assert infos[agent_id]["reward_breakdown"].qos_satisfaction == pytest.approx(expected_qos)
