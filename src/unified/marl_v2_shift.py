"""Can a trained controller follow a change in traffic that a fixed hand-tuned rule cannot?

Controllers are trained or tuned on ordinary Milan traffic (days 1-40), then scored on the
held-out week (days 45-51) with NO retraining after the traffic mix changes:
  baseline       as recorded
  growth_6g      eMBB x1.3, URLLC x2, mMTC x3
  massive_iot    mMTC x10
The forecast is scaled with the demand (the forecaster itself is not retrained).

Controllers:  equal split | expert tuned on baseline | expert re-tuned on the shifted traffic
              (a human re-doing the work) | shared-actor MARL trained on baseline, with and
              without the forecast (3 seeds each).

Run: python src/unified/marl_v2_shift.py <ddpg_fc|ddpg_nofc|experts>
Writes reports/unified/marl_v2_shift_<variant>.json
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from unified import env as envmod  # noqa: E402
from unified import marl_v2 as m  # noqa: E402
from unified.agents import SharedMADDPG, TrainConfig  # noqa: E402
from unified.env import equal_split, headroom, run_policy  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = {"baseline": (1.0, 1.0, 1.0), "growth_6g": (1.3, 2.0, 3.0), "massive_iot": (1.0, 1.0, 10.0)}
U_URLLC, U_MMTC = (0.5, 0.4, 0.3, 0.25, 0.2, 0.15), (0.65, 0.5, 0.4, 0.3, 0.25, 0.2)


def scaled(D, F, mult):
    k = np.array(mult, np.float32)
    return D * k, F * k


def week(env, pol):
    return [run_policy(env, pol, 144 * k + 1, m.DAY)["mean_reward"] for k in range(7)]


def tune(D, F, source):
    train = m.make_env(D, F, 0, 41 * m.DAY)
    best, best_s = None, -1
    for u in itertools.product(U_URLLC, U_MMTC):
        envmod.HEADROOM_UTILISATION = (0.65, *u)
        s = float(np.mean([run_policy(train, lambda e, o: headroom(e, o, source), d * m.DAY, m.DAY - 2)["mean_reward"] for d in range(1, 41, 3)]))
        if s > best_s:
            best, best_s = u, s
    envmod.HEADROOM_UTILISATION = (0.65, 0.5, 0.65)
    return best


def expert_policy(source, util):
    def pol(e, o):
        envmod.HEADROOM_UTILISATION = (0.65, *util)
        try:
            return headroom(e, o, source)
        finally:
            envmod.HEADROOM_UTILISATION = (0.65, 0.5, 0.65)    # never leak tuned thresholds into other policies
    return pol


def main(variant):
    torch.set_num_threads(4)
    D, F = m.build_data()
    base, end = m.TEST_DAYS[0] * m.DAY - 1, m.TEST_DAYS[1] * m.DAY
    out = {"variant": variant, "scenarios": {k: list(v) for k, v in SCENARIOS.items()}}
    if variant == "experts":
        tuned_base = {s: tune(D, F, s) for s in ("forecast", "current")}
        out["tuned_on_baseline"] = {k: list(v) for k, v in tuned_base.items()}
        res = {}
        for name, mult in SCENARIOS.items():
            Ds, Fs = scaled(D, F, mult)
            test = m.make_env(Ds, Fs, base, end)
            row = {"equal_split": week(test, lambda e, o: equal_split(e, o))}
            for s in ("forecast", "current"):
                row[f"expert_{s}_tuned_on_baseline"] = week(test, expert_policy(s, tuned_base[s]))
                retuned = tune(Ds, Fs, s)
                row[f"expert_{s}_retuned"] = week(test, expert_policy(s, retuned))
                row[f"expert_{s}_retuned_util"] = list(retuned)
            row["expert_oracle_default"] = week(test, lambda e, o: headroom(e, o, "oracle"))
            res[name] = {k: ([round(x, 4) for x in v] if k.endswith("_util") is False else v) for k, v in row.items()}
            print(name, {k: round(float(np.mean(v)), 4) for k, v in row.items() if not k.endswith("_util")}, flush=True)
        out["results"] = res
    else:
        use_fc = variant == "ddpg_fc"
        runs = []
        for seed in m.SEEDS:
            t0 = time.time()
            env = m.make_env(D, F if use_fc else None, m.TRAIN_DAYS[0] * m.DAY, m.TRAIN_DAYS[1] * m.DAY, seed)
            agent = SharedMADDPG(env.N, env.obs_dim, env.nbr_matrix, TrainConfig(episodes=m.EPISODES, seed=seed))
            agent.train(env, log_every=500)
            r = {"seed": seed, "train_seconds": round(time.time() - t0)}
            for name, mult in SCENARIOS.items():
                Ds, Fs = scaled(D, F, mult)
                r[name] = [round(x, 4) for x in week(m.make_env(Ds, Fs if use_fc else None, base, end), agent.policy)]
            runs.append(r)
            print(variant, "seed", seed, {k: round(float(np.mean(v)), 4) for k, v in r.items() if k in SCENARIOS}, flush=True)
        out["runs"] = runs
    (ROOT / "reports" / "unified" / f"marl_v2_shift_{variant}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", variant)


if __name__ == "__main__":
    main(sys.argv[1])
