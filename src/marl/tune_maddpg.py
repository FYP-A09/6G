"""Small honest hyperparameter search for MADDPG on the full B5G release.

Protocol (no test-set peeking): train each config on samples [0, TRAIN),
select on a validation block [TRAIN, TRAIN+50), then report the selected config
once on a fresh test block [TEST_START, TEST_START+50) next to the baselines.
Run directly: python src/marl/tune_maddpg.py
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from marl.b5g_env import B5GSlicingEnv  # noqa: E402
from marl.evaluation import _evaluate_policy, equal_split, heuristic_action  # noqa: E402
from marl.maddpg import MADDPGTrainer, TrainingConfig  # noqa: E402

ROOT = r"E:\FYP DATA\6G\data\raw\b5g_slicing\2.2.1-10kprocessed"
TRAIN, VAL_SAMPLES = 1000, 50
TEST_START, TEST_SAMPLES = 3000, 50
OUT = Path("reports/krish_marl_tuned")


def evaluate(trainer: MADDPGTrainer, start: int, limit: int):
    def policy(observation):
        return trainer._action_dict(trainer.select_action(observation))

    policy.batch = trainer.select_actions
    env = B5GSlicingEnv(data_root=ROOT, load_topology=False)
    return _evaluate_policy(env, "trained_actor", policy, limit, start_index=start)


def main() -> None:
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    grid = list(itertools.product((0.05, 0.2), (1e-3, 1e-4), (1.0, 0.01)))
    results = []
    for noise, actor_lr, tau in grid:
        config = TrainingConfig(
            episodes=TRAIN, exploration_noise=noise, actor_learning_rate=actor_lr, target_tau=tau,
        )
        started = time.perf_counter()
        trainer = MADDPGTrainer(config)
        trainer.train(B5GSlicingEnv(data_root=ROOT, load_topology=False))
        validation = evaluate(trainer, TRAIN, VAL_SAMPLES)
        record = {
            "exploration_noise": noise, "actor_lr": actor_lr, "target_tau": tau,
            "validation_mean_reward": validation.mean_reward,
            "validation_utilization": validation.mean_utilization,
            "seconds": time.perf_counter() - started,
        }
        results.append((record, trainer))
        print(json.dumps(record), flush=True)

    best_record, best_trainer = max(results, key=lambda item: item[0]["validation_mean_reward"])
    test = evaluate(best_trainer, TEST_START, TEST_SAMPLES)
    baselines = []
    for name, policy in (("equal_split", equal_split), ("heuristic", heuristic_action)):
        env = B5GSlicingEnv(data_root=ROOT, load_topology=False)
        baselines.append(asdict(_evaluate_policy(env, name, policy, TEST_SAMPLES, start_index=TEST_START)))
    report = {
        "protocol": f"train samples [0,{TRAIN}), validate [{TRAIN},{TRAIN + VAL_SAMPLES}), test [{TEST_START},{TEST_START + TEST_SAMPLES})",
        "configs": [record for record, _ in results],
        "selected": best_record,
        "test_trained_actor": asdict(test),
        "test_baselines": baselines,
    }
    (OUT / "tuning_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("SELECTED", json.dumps(best_record))
    print("TEST trained_actor", test.mean_reward, "util", test.mean_utilization)
    for baseline in baselines:
        print("TEST", baseline["policy"], baseline["mean_reward"])


if __name__ == "__main__":
    main()
