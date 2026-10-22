"""Where does the trained MARL controller's gain over the expert rules come from?

Retrains one seed of the plain shared-actor MADDPG with the forecast (same data, budget and
seed as marl_v2.py), then on the held-out week compares it with the expert rules per slice:
SLA met, share of demand served, mean allocation, idle share. A gain that comes from meeting
more SLAs at the same service level is a real improvement; a gain that comes from starving a
slice the reward barely weighs would be reward gaming, and is reported as such.

Run: python src/unified/marl_v2_diagnose.py  ->  reports/unified/marl_v2_diagnosis.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from unified import marl_v2 as m  # noqa: E402
from unified.agents import SharedMADDPG, TrainConfig  # noqa: E402
from unified.env import SLICES, headroom, equal_split  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def rollout(env, policy, steps_per_day=144, days=7):
    ok, served, demand, share, rew = [], [], [], [], []
    for k in range(days):
        obs = env.reset(start=144 * k + 1)
        env.t_end = min(144 * k + 1 + steps_per_day, env.T - 1)
        done = False
        while not done:
            obs, r, _, done, info = env.step(policy(env, obs))
            ok.append(info["sla_ok"]); served.append(info["served"]); demand.append(info["demand_mbps"])
            share.append(env.prev_action.copy()); rew.append(r.mean())
    ok, served, demand, share = map(np.array, (ok, served, demand, share))
    return {"sla_ok": dict(zip(SLICES, ok.mean(axis=(0, 1)).round(4).tolist())),
            "all_slices_ok": round(float(ok.all(axis=2).mean()), 4),
            "served_fraction": round(float(served.sum() / demand.sum()), 4),
            "served_fraction_by_slice": dict(zip(SLICES, (served.sum(axis=(0, 1)) / demand.sum(axis=(0, 1))).round(4).tolist())),
            "mean_share": dict(zip(SLICES + ("idle",), share.mean(axis=(0, 1)).round(4).tolist())),
            "mean_reward": round(float(np.mean(rew)), 4)}


def main():
    torch.set_num_threads(4)
    D, F = m.build_data()
    base, end = m.TEST_DAYS[0] * m.DAY - 1, m.TEST_DAYS[1] * m.DAY
    env = m.make_env(D, F, m.TRAIN_DAYS[0] * m.DAY, m.TRAIN_DAYS[1] * m.DAY, 0)
    test = m.make_env(D, F, base, end)
    cfg = TrainConfig(episodes=m.EPISODES, seed=0)
    agent = SharedMADDPG(env.N, env.obs_dim, env.nbr_matrix, cfg)
    agent.train(env, log_every=500)
    out = {"marl_ddpg_fc_seed0": rollout(test, agent.policy),
           "expert_forecast": rollout(test, lambda e, o: headroom(e, o, "forecast")),
           "expert_current": rollout(test, lambda e, o: headroom(e, o, "current")),
           "expert_oracle": rollout(test, lambda e, o: headroom(e, o, "oracle")),
           "equal_split": rollout(test, lambda e, o: equal_split(e, o))}
    (ROOT / "reports" / "unified" / "marl_v2_diagnosis.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for k, v in out.items():
        print(k, v["mean_reward"], "all-ok", v["all_slices_ok"], "SLA", v["sla_ok"], "served", v["served_fraction"], "share", v["mean_share"])


if __name__ == "__main__":
    main()
