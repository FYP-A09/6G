"""Do the controller rankings survive a latency curve measured in a packet-level 5G simulator?

All scored on the held-out week with the Simu5G-shaped QoS curve (qos_alt.py: flat delay, then a cliff at capacity):
  - equal split, the untuned expert rule, and the expert rule tuned on the training days under this curve
    (a wider grid, since the optimum moves towards much less headroom)
  - controllers trained on the default curve and run here without retraining: MAPPO and DDPG-MARL actors
  - MAPPO retrained under this curve (marl_v2_ppo.py with QOS=simu5g), if those runs exist

Run: python src/unified/marl_v2_simu5g.py -> reports/unified/marl_v2_simu5g.json
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from unified import marl_v2 as m  # noqa: E402
from unified import marl_v2_ppo as pp  # noqa: E402  (imported before the curve is switched, so SUFFIX is empty)
from unified import env as envmod  # noqa: E402
from unified import qos_alt  # noqa: E402
from unified.agents import SharedMADDPG, TrainConfig  # noqa: E402
from unified.env import equal_split, headroom, run_policy  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CK = ROOT / "data" / "processed" / "unified"
GRID = (0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4)


def week(env, pol):
    return [run_policy(env, pol, 144 * i + 1, m.DAY)["mean_reward"] for i in range(7)]


def main():
    info = qos_alt.apply()
    D, F = m.build_data()
    base, end = m.TEST_DAYS[0] * m.DAY - 1, m.TEST_DAYS[1] * m.DAY
    test, train = m.make_env(D, F, base, end), m.make_env(D, F, 0, 41 * m.DAY)
    out = {"curve": info, "controllers": {}}
    rec = lambda name, per: out["controllers"].__setitem__(name, {"test_per_day": [round(x, 4) for x in per], "test_mean": round(float(np.mean(per)), 4)})
    rec("Equal split", week(test, lambda e, o: equal_split(e, o)))
    for source in ("current", "forecast"):
        rec(f"Expert rule, untuned, {source}", week(test, lambda e, o, s=source: headroom(e, o, s)))
        best, best_s = None, -1
        for u in itertools.product(GRID, GRID):
            envmod.HEADROOM_UTILISATION = (0.65, *u)
            sc = float(np.mean([run_policy(train, lambda e, o, s=source: headroom(e, o, s), d * m.DAY, m.DAY - 2)["mean_reward"] for d in range(1, 41, 3)]))
            if sc > best_s:
                best, best_s = u, sc
        envmod.HEADROOM_UTILISATION = (0.65, *best)
        rec(f"Expert rule, tuned under this curve, {source}", week(test, lambda e, o, s=source: headroom(e, o, s)))
        out["controllers"][f"Expert rule, tuned under this curve, {source}"]["targets_urllc_mmtc"] = list(best)
        envmod.HEADROOM_UTILISATION = (0.65, 0.5, 0.65)
    rec("Expert rule on the true next demand, untuned", week(test, lambda e, o: headroom(e, o, "oracle")))
    for f in sorted(CK.glob("marl_v2_ppo_*.pt")):
        mode, seed = f.stem.replace("marl_v2_ppo_", "").replace("_simu5g", "").split("_")
        ck = torch.load(f, weights_only=False)
        a = pp.PPO(ck["n"], ck["f"], test.nbr_matrix, mode, int(seed))
        a.actor.load_state_dict(ck["actor"])
        tag = "retrained under this curve" if "simu5g" in f.stem else "trained on the default curve, not retrained"
        rec(f"{mode.upper()} seed {seed}, {tag}", week(test, a.policy))
    for seed in m.SEEDS:
        f = CK / f"marl_v2_best_seed{seed}.pt"
        if f.exists():
            ck = torch.load(f, weights_only=False)
            a = SharedMADDPG(ck["n"], ck["f"], test.nbr_matrix, TrainConfig())
            a.actor.load_state_dict(ck["actor"])
            rec(f"DDPG-MARL seed {seed}, trained on the default curve, not retrained", week(test, a.policy))
    for k, v in out["controllers"].items():
        print(f"{v['test_mean']:.4f}  {k}", flush=True)
    (ROOT / "reports" / "unified" / "marl_v2_simu5g.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
