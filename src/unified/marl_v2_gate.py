"""A Digital Twin approve/reject gate in the unified environment, and what it is worth.

Before a cell's allocation is applied, the twin predicts its outcome for the next interval with the environment's
own QoS model and the TGNN forecast of that interval's demand (not the true demand), for the proposed allocation and
for a safe fallback (the expert rule tuned on the training days). The proposal is approved if its predicted reward is
at least the fallback's predicted reward minus a margin; otherwise the fallback is applied. The gate is per cell.

It is measured on the held-out week with controllers whose proposals are sometimes bad, as when a learning agent
explores online: the trained MARL actor with logit noise sigma in {0, 0.5, 1.0, 2.0}, and an equal split. The
gate's effect is the realised reward with the gate minus without it, and how often it approves.

Run: python src/unified/marl_v2_gate.py -> reports/unified/marl_v2_gate.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from unified import env as envmod  # noqa: E402
from unified import marl_v2 as m  # noqa: E402
from unified.agents import SharedMADDPG, TrainConfig  # noqa: E402
from unified.env import ModelConfig, equal_split, headroom, qos  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CK = ROOT / "data" / "processed" / "unified"
TUNED = (0.65, 0.4, 0.4)
MARGIN = 0.0


def predicted_reward(env, action, demand_hat):
    cfg = env.cfg
    a = np.clip(np.asarray(action, float), 0, None)
    a = a / np.maximum(a.sum(axis=1, keepdims=True), 1e-9)
    cap = env.capacity_at(env.t + 1)
    q = qos(demand_hat, a[:, :3] * cap[:, None], cfg)
    sla = (q["sla_ok"] * np.array(cfg.slice_weight)).sum(axis=1)
    total = demand_hat.sum(axis=1)
    served = np.where(total > 0, q["served"].sum(axis=1) / np.maximum(total, 1e-9), 1.0)
    churn = 0.5 * np.abs(a - env.prev_action).sum(axis=1)
    return cfg.w_sla * sla + cfg.w_served * served - cfg.w_churn * churn


def run_week(env, propose, gated, margin=MARGIN):
    per_day, approved, steps = [], 0, 0
    for k in range(7):
        env.reset(start=144 * k + 1)
        env.t_end = min(144 * k + 1 + m.DAY, env.T - 1)
        done, rs = False, []
        while not done:
            cand = propose(env)
            act = cand
            if gated:
                envmod.HEADROOM_UTILISATION = TUNED
                fb = headroom(env, None, "forecast")
                envmod.HEADROOM_UTILISATION = (0.65, 0.5, 0.65)
                dem = env.forecast[env.t].astype(float)
                ok = predicted_reward(env, cand, dem) >= predicted_reward(env, fb, dem) - margin
                act = np.where(ok[:, None], cand, fb)
                approved += int(ok.sum()); steps += len(ok)
            _, r, _, done, _ = env.step(act)
            rs.append(r.mean())
        per_day.append(float(np.mean(rs)))
    return per_day, (approved / steps if steps else None)


def main():
    torch.set_num_threads(4)
    D, F = m.build_data()
    base, end = m.TEST_DAYS[0] * m.DAY - 1, m.TEST_DAYS[1] * m.DAY
    out = {"margin": MARGIN, "fallback": "expert rule tuned on the training days, with the forecast", "controllers": {}}
    actors = []
    test = m.make_env(D, F, base, end)
    for seed in m.SEEDS:
        ck = torch.load(CK / f"marl_v2_best_seed{seed}.pt", weights_only=False)
        a = SharedMADDPG(ck["n"], ck["f"], test.nbr_matrix, TrainConfig())
        a.actor.load_state_dict(ck["actor"])
        actors.append(a)
    rng = np.random.default_rng(0)

    def marl(sigma, a):
        def propose(env):
            obs = env._obs()
            with torch.no_grad():
                lg = a.actor(torch.from_numpy(obs))
            if sigma:
                lg = lg + sigma * torch.from_numpy(rng.standard_normal(lg.shape).astype(np.float32))
            return torch.softmax(lg, dim=-1).numpy()
        return propose

    rows = [(f"MARL, sigma {s}", [marl(s, a) for a in actors]) for s in (0.0, 0.5, 1.0, 2.0)]
    rows.append(("Equal split", [lambda env: equal_split(env, None)]))
    for name, proposers in rows:
        res = {"without_gate": [], "with_gate": [], "approval_rate": []}
        for pr in proposers:
            wo, _ = run_week(m.make_env(D, F, base, end), pr, False)
            wi, ap = run_week(m.make_env(D, F, base, end), pr, True)
            res["without_gate"].append(float(np.mean(wo))); res["with_gate"].append(float(np.mean(wi))); res["approval_rate"].append(ap)
        out["controllers"][name] = {k: [round(x, 4) for x in v] for k, v in res.items()}
        print(name, "without", round(float(np.mean(res["without_gate"])), 4), "with gate", round(float(np.mean(res["with_gate"])), 4), "approved", round(float(np.mean(res["approval_rate"])), 3), flush=True)
    fb = []
    envmod.HEADROOM_UTILISATION = TUNED
    fb, _ = run_week(m.make_env(D, F, base, end), lambda env: headroom(env, None, "forecast"), False)
    envmod.HEADROOM_UTILISATION = (0.65, 0.5, 0.65)
    out["fallback_alone"] = round(float(np.mean(fb)), 4)
    (ROOT / "reports" / "unified" / "marl_v2_gate.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
