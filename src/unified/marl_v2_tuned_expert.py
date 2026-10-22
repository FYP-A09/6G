"""A fair expert baseline: tune the headroom rule's URLLC and mMTC utilisation targets on the
training days (1-40) by grid search, with and without the forecast, then score it on the held-out
week. If the learned MARL controller only matched this, its gain would be a tuning effect.

Run: python src/unified/marl_v2_tuned_expert.py -> reports/unified/marl_v2_tuned_expert.json
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from unified import env as envmod  # noqa: E402
from unified import marl_v2 as m  # noqa: E402
from unified.env import headroom, run_policy  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
U_URLLC, U_MMTC = (0.5, 0.4, 0.3, 0.25, 0.2, 0.15), (0.65, 0.5, 0.4, 0.3, 0.25, 0.2)
TRAIN_SAMPLE_DAYS = list(range(1, 41, 3))     # 14 training days


def score(env, source, util, starts, steps):
    envmod.HEADROOM_UTILISATION = (0.65, *util)
    pol = lambda e, o: headroom(e, o, source)
    return [run_policy(env, pol, s, steps)["mean_reward"] for s in starts]


def main():
    D, F = m.build_data()
    train = m.make_env(D, F, 0, 41 * m.DAY)
    test = m.make_env(D, F, m.TEST_DAYS[0] * m.DAY - 1, m.TEST_DAYS[1] * m.DAY)
    out = {"grid": {"urllc": U_URLLC, "mmtc": U_MMTC}, "train_days_sampled": TRAIN_SAMPLE_DAYS}
    for source in ("forecast", "current"):
        grid = {}
        for u in itertools.product(U_URLLC, U_MMTC):
            grid[u] = float(np.mean(score(train, source, u, [d * m.DAY for d in TRAIN_SAMPLE_DAYS], m.DAY - 2)))
        best = max(grid, key=grid.get)
        per_day = score(test, source, best, [144 * k + 1 for k in range(7)], m.DAY)
        default = score(test, source, (0.5, 0.65), [144 * k + 1 for k in range(7)], m.DAY)
        out[source] = {"best_utilisation_urllc_mmtc": best, "train_reward_best": round(grid[best], 4), "train_reward_default": round(grid[(0.5, 0.65)], 4),
                       "test_per_day": [round(x, 4) for x in per_day], "test_mean": round(float(np.mean(per_day)), 4),
                       "default_test_mean": round(float(np.mean(default)), 4)}
        print(source, out[source], flush=True)
    envmod.HEADROOM_UTILISATION = (0.65, 0.5, 0.65)
    (ROOT / "reports" / "unified" / "marl_v2_tuned_expert.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
