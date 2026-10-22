"""A much stronger rule baseline: expert headroom with utilisation targets tuned per time of day and per cell class.

The simple tuned expert has two numbers for the whole network. Here every (4-hour block, cell class) pair has its
own URLLC and mMTC utilisation target, chosen by grid search on the training days only (cell classes = terciles of
each cell's mean demand on the training days). One rollout per candidate pair is enough: the per-step, per-cell
reward is split by block and class and each group keeps the pair that maximises its own reward. The result is
scored on the held-out week with and without the forecast.

Run: python src/unified/marl_v2_tuned_expert_rich.py -> reports/unified/marl_v2_tuned_expert_rich.json
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

ROOT = Path(__file__).resolve().parents[2]
U_URLLC, U_MMTC = (0.5, 0.4, 0.3, 0.25, 0.2, 0.15), (0.65, 0.5, 0.4, 0.3, 0.25, 0.2)
BLOCKS, CLASSES = 6, 3
DEFAULT = (0.65, 0.5, 0.65)


def rollout(env, source, util_for_step, starts, steps, offset):
    """Per-step, per-cell rewards and the absolute step index, for a policy whose utilisation varies by step and cell."""
    rewards, abs_t = [], []
    for s in starts:
        env.reset(start=s)
        env.t_end = min(s + steps, env.T - 1)
        done = False
        while not done:
            u = util_for_step((env.t + offset) % m.DAY)
            envmod.HEADROOM_UTILISATION = np.column_stack([np.full(env.N, 0.65), u[:, 0], u[:, 1]])
            a = envmod.headroom(env, None, source)
            _, r, _, done, _ = env.step(a)
            rewards.append(r)
            abs_t.append(env.t - 1 + offset)
    envmod.HEADROOM_UTILISATION = DEFAULT
    return np.array(rewards), np.array(abs_t)


def tune(env, source, cls):
    """Best (urllc, mmtc) target for every (block, class) group, from rollouts on the training days."""
    combos = list(itertools.product(U_URLLC, U_MMTC))
    starts = [d * m.DAY for d in range(1, 41, 3)]
    score = np.zeros((len(combos), BLOCKS, CLASSES))
    count = np.zeros((BLOCKS, CLASSES))
    for k, (a, b) in enumerate(combos):
        r, t = rollout(env, source, lambda tod, a=a, b=b: np.tile([a, b], (env.N, 1)), starts, m.DAY - 2, 0)
        block = (t % m.DAY) * BLOCKS // m.DAY
        for bl in range(BLOCKS):
            for c in range(CLASSES):
                sel = (block == bl)
                if sel.any() and (cls == c).any():
                    score[k, bl, c] = r[sel][:, cls == c].mean()
                    count[bl, c] = 1
    best = score.argmax(axis=0)
    return {(bl, c): combos[best[bl, c]] for bl in range(BLOCKS) for c in range(CLASSES)}


def main():
    D, F = sc.build_data100()
    base, end = m.TEST_DAYS[0] * m.DAY - 1, m.TEST_DAYS[1] * m.DAY
    out = {"blocks": BLOCKS, "classes": CLASSES}
    for size in (3, 5, 10):
        n = size * size
        train = sc.make_env(D, F, size, 0, 41 * m.DAY)
        test = sc.make_env(D, F, size, base, end)
        mean_dem = D[m.DAY:41 * m.DAY].sum(axis=2)
        full = sc.block_cells(10)
        idx = [full.index(c) for c in sc.block_cells(size)]
        md = mean_dem[:, idx].mean(axis=0)
        cls = np.digitize(md, np.quantile(md, [1 / 3, 2 / 3]))
        res = {}
        for source in ("forecast", "current"):
            table = tune(train, source, cls)

            def util_for(tod, table=table):
                bl = tod * BLOCKS // m.DAY
                return np.array([table[(bl, c)] for c in cls])

            per = []
            for k in range(7):
                r, _ = rollout(test, source, util_for, [144 * k + 1], m.DAY, base)
                per.append(float(r.mean()))
            res[source] = {"test_per_day": [round(x, 4) for x in per], "test_mean": round(float(np.mean(per)), 4)}
            print(n, source, res[source]["test_mean"], flush=True)
        out[str(n)] = res
        (ROOT / "reports" / "unified" / "marl_v2_tuned_expert_rich.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
