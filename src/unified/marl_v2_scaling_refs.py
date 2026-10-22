"""Reference policies for the scaling study: equal split and the expert rule (default and tuned on the
training days, with and without the forecast), at 9 / 25 / 100 cells on the held-out week.

Run: python src/unified/marl_v2_scaling_refs.py -> reports/unified/marl_v2_scaling_refs.json
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
from unified import marl_v2_scaling as sc  # noqa: E402
from unified.env import equal_split, headroom, run_policy  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
U_URLLC, U_MMTC = (0.5, 0.4, 0.3, 0.25, 0.2, 0.15), (0.65, 0.5, 0.4, 0.3, 0.25, 0.2)


def main():
    D, F = sc.build_data100()
    base, end = m.TEST_DAYS[0] * m.DAY - 1, m.TEST_DAYS[1] * m.DAY
    out = {}
    for size in (3, 5, 10):
        n = size * size
        train, test = sc.make_env(D, F, size, 0, 41 * m.DAY), sc.make_env(D, F, size, base, end)
        week = lambda pol: [run_policy(test, pol, 144 * k + 1, m.DAY)["mean_reward"] for k in range(7)]
        row = {"equal_split": week(lambda e, o: equal_split(e, o)), "expert_oracle": week(lambda e, o: headroom(e, o, "oracle"))}
        for source in ("current", "forecast"):
            row[f"expert_{source}_default"] = week(lambda e, o: headroom(e, o, source))
            best, best_s = None, -1
            for u in itertools.product(U_URLLC, U_MMTC):
                envmod.HEADROOM_UTILISATION = (0.65, *u)
                s = float(np.mean([run_policy(train, lambda e, o: headroom(e, o, source), d * m.DAY, m.DAY - 2)["mean_reward"] for d in range(1, 41, 3)]))
                if s > best_s:
                    best, best_s = u, s
            envmod.HEADROOM_UTILISATION = (0.65, *best)
            row[f"expert_{source}_tuned"] = week(lambda e, o: headroom(e, o, source))
            row[f"expert_{source}_tuned_util"] = list(best)
            envmod.HEADROOM_UTILISATION = (0.65, 0.5, 0.65)
        out[str(n)] = {k: ([round(x, 4) for x in v] if not k.endswith("_util") else v) for k, v in row.items()}
        print(n, {k: round(float(np.mean(v)), 4) for k, v in row.items() if not k.endswith("_util")}, flush=True)
    (ROOT / "reports" / "unified" / "marl_v2_scaling_refs.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
