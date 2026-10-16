"""Experiments on the unified environment, answering the supervisor's questions.

1. Forecast quality: TGNN vs persistence on held-out days.
2. Does prediction help allocation? Expert headroom policy fed current demand,
   the TGNN forecast, or the true next demand; and MARL with no forecast / TGNN / oracle.
3. Why MARL and not a single agent? Single centralised DDPG vs shared-parameter
   MADDPG at N = 9, 25, 49, 100 cells, 3 seeds each, same observations and update budget.

Split: days 1-5 (intervals 0-719) train, days 6-7 (720-1007) test. Results are
written to reports/unified/results.json after every run.
Run: python src/unified/experiments.py [--quick]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from unified.agents import SharedMADDPG, SingleAgentDDPG, TrainConfig, n_params  # noqa: E402
from unified.data import block_cells, grid_neighbours, load_block_activity  # noqa: E402
from unified.env import (ModelConfig, UnifiedSlicingEnv, demand_scale, equal_split,  # noqa: E402
                         headroom, proportional, run_policy)
from unified.forecast import forecast_errors, train_forecaster  # noqa: E402

MILAN = r"E:\FYP DATA\6G\data\raw\milan_telecom_italia_full\data"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "unified"
CACHE = ROOT / "data" / "processed" / "unified" / "tgnn_forecast.npz"
TRAIN_END, EP_LEN = 720, 36
SIZES = (3, 5, 7, 10)
SEEDS = (0, 1, 2)
# Single-agent learning rate chosen by a 5-config sweep on a held-out training day
# (reports/unified/single_agent_sweep.json); MARL keeps the defaults.
SINGLE_AGENT_ACTOR_LR = 1e-4


def save(results: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


def make_env(demand, cells_full, size, forecast=None, seed=0):
    cells = block_cells(size)
    idx = [cells_full.index(c) for c in cells]
    d = demand[:, idx, :]
    f = None if forecast is None else forecast[:, idx, :]
    return UnifiedSlicingEnv(d, grid_neighbours(cells), ModelConfig(), episode_len=EP_LEN,
                             start_range=(0, TRAIN_END - EP_LEN - 1), forecast=f, seed=seed)


def evaluate(env, policy) -> dict:
    return run_policy(env, policy, TRAIN_END, env.T - 1 - TRAIN_END)


def run_agent(cls, env, seed, episodes) -> dict:
    cfg = TrainConfig(episodes=episodes, seed=seed)
    if cls is SingleAgentDDPG:
        cfg.actor_lr = SINGLE_AGENT_ACTOR_LR
    agent = cls(env.N, env.obs_dim, env.nbr_matrix, cfg)
    hist = agent.train(env)
    m = evaluate(env, agent.policy)
    obs = env.reset(start=TRAIN_END)
    t0 = time.perf_counter()
    for _ in range(50):
        agent.act(obs, 0.0)
    m.update({"train_seconds": hist["train_seconds"], "updates": hist["updates"],
              "learning_curve": [round(x, 4) for x in hist["episode_reward"]],
              "actor_params": n_params(agent.actor), "critic_params": n_params(agent.critic),
              "decision_ms": (time.perf_counter() - t0) / 50 * 1000,
              "actor_input_dim": agent.actor[0].in_features, "actor_output_dim": agent.actor[-1].out_features})
    return m


def main(quick: bool = False) -> None:
    torch.set_num_threads(4)
    episodes = 40 if quick else 300
    seeds = SEEDS[:1] if quick else SEEDS
    data = load_block_activity(MILAN)
    cells_full = [int(c) for c in data["cells"]]
    cfg = ModelConfig()
    scale = demand_scale(data["activity"][:TRAIN_END], cfg)
    demand = data["activity"] * scale
    results = {"split": {"train_intervals": [0, TRAIN_END], "test_intervals": [TRAIN_END, int(demand.shape[0])]},
               "demand_scale_mbps_per_activity_unit": scale.tolist(), "model_config": vars(cfg)}

    # 1. forecaster (trained once on the full 100-cell block, training days only)
    if CACHE.exists():
        forecast = np.load(CACHE)["forecast"]
        fc_hist = None
    else:
        out = train_forecaster(demand, grid_neighbours(cells_full), TRAIN_END, epochs=6 if quick else 30)
        forecast, fc_hist = out["forecast"], out["val_history"]
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE, forecast=forecast)
    results["forecast"] = forecast_errors(demand, forecast, TRAIN_END, demand.shape[0])
    results["forecast"]["val_history"] = fc_hist
    print("forecast:", results["forecast"], flush=True)
    save(results)

    # 2a. reference policies at every size
    results["heuristics"] = {}
    for size in SIZES:
        env = make_env(demand, cells_full, size, forecast)
        results["heuristics"][str(size * size)] = {
            "equal_split": evaluate(env, equal_split),
            "proportional_current": evaluate(env, lambda e, o: proportional(e, o, "current")),
            "headroom_current": evaluate(env, lambda e, o: headroom(e, o, "current")),
            "headroom_tgnn": evaluate(env, lambda e, o: headroom(e, o, "forecast")),
            "headroom_oracle": evaluate(env, lambda e, o: headroom(e, o, "oracle")),
        }
    print("heuristics done", flush=True)
    save(results)

    # 3. single agent vs MARL across sizes (both observe the TGNN forecast)
    results["scaling"] = {}
    for size in SIZES:
        n = str(size * size)
        results["scaling"][n] = {"single_agent": [], "marl": []}
        for seed in seeds:
            for key, cls in (("single_agent", SingleAgentDDPG), ("marl", SharedMADDPG)):
                env = make_env(demand, cells_full, size, forecast, seed=seed)
                m = run_agent(cls, env, seed, episodes)
                results["scaling"][n][key].append(m)
                print(f"N={n} {key} seed={seed}: test reward {m['mean_reward']:.4f} sla {m['sla_score']:.4f} "
                      f"({m['train_seconds']:.0f}s)", flush=True)
                save(results)

    # 2b. MARL forecast ablation at N=25 (TGNN runs are reused from the scaling sweep)
    oracle = np.concatenate([demand[1:], demand[-1:]], axis=0)
    results["ablation_marl_n25"] = {"tgnn": results["scaling"]["25"]["marl"], "none": [], "oracle": []}
    for seed in seeds:
        for name, fc in (("none", None), ("oracle", oracle)):
            env = make_env(demand, cells_full, 5, fc, seed=seed)
            m = run_agent(SharedMADDPG, env, seed, episodes)
            results["ablation_marl_n25"][name].append(m)
            print(f"ablation {name} seed={seed}: test reward {m['mean_reward']:.4f}", flush=True)
            save(results)
    print("done", flush=True)


def finish() -> None:
    """Complete an interrupted run from results.json: missing MARL seeds, the single-agent arm
    re-run with the tuned learning rate at every size, and the forecast ablation. Saves after every run."""
    torch.set_num_threads(4)
    results = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    data = load_block_activity(MILAN)
    cells_full = [int(c) for c in data["cells"]]
    demand = data["activity"] * demand_scale(data["activity"][:TRAIN_END], ModelConfig())
    forecast = np.load(CACHE)["forecast"]
    results["single_agent_actor_lr"] = SINGLE_AGENT_ACTOR_LR
    for size in SIZES:
        n = str(size * size)
        block = results["scaling"].setdefault(n, {"single_agent": [], "marl": []})
        for seed in SEEDS[len(block["marl"]):]:
            m = run_agent(SharedMADDPG, make_env(demand, cells_full, size, forecast, seed=seed), seed, 300)
            block["marl"].append(m)
            print(f"N={n} marl seed={seed}: test reward {m['mean_reward']:.4f}", flush=True)
            save(results)
        tuned = [r for r in block["single_agent"] if r.get("actor_lr") == SINGLE_AGENT_ACTOR_LR]
        for seed in SEEDS[len(tuned):]:
            m = run_agent(SingleAgentDDPG, make_env(demand, cells_full, size, forecast, seed=seed), seed, 300)
            m["actor_lr"] = SINGLE_AGENT_ACTOR_LR
            tuned.append(m)
            block["single_agent"] = tuned
            print(f"N={n} single_agent (tuned) seed={seed}: test reward {m['mean_reward']:.4f}", flush=True)
            save(results)
    oracle = np.concatenate([demand[1:], demand[-1:]], axis=0)
    abl = results.setdefault("ablation_marl_n25", {"none": [], "oracle": []})
    abl["tgnn"] = results["scaling"]["25"]["marl"]
    for name, fc in (("none", None), ("oracle", oracle)):
        for seed in SEEDS[len(abl.setdefault(name, [])):]:
            m = run_agent(SharedMADDPG, make_env(demand, cells_full, 5, fc, seed=seed), seed, 300)
            abl[name].append(m)
            print(f"ablation {name} seed={seed}: test reward {m['mean_reward']:.4f}", flush=True)
            save(results)
    print("finished", flush=True)


if __name__ == "__main__":
    if "--finish" in sys.argv:
        finish()
    else:
        main(quick="--quick" in sys.argv)
