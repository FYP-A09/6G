"""MARL trained across traffic mixes, so a change in the mix does not break it.

marl_v2_shift.py showed that a controller trained on ordinary traffic loses to a hand-tuned rule once the traffic mix
changes. The fix is the standard one: randomise the mix during training. At the start of every episode each slice's
demand (and its forecast) is multiplied by a random factor, eMBB in [1, 1.5], URLLC in [1, 3], mMTC in [1, 10] (log-uniform),
with probability 0.5 each and otherwise the factor is 1, so ordinary traffic stays well covered. The rest is the
long protocol of marl_v2_best.py (25 cells, 4000 episodes, checkpoint chosen on validation days, which are scored on the
recorded traffic AND on both shifted mixes, so the checkpoint is not chosen for one mix only).

Evaluation: the held-out week under recorded traffic, the 6G-growth mix and the massive-IoT mix, against the expert rule
tuned on the training days and the same rule re-tuned on each mix (a human redoing the work).

Run: python src/unified/marl_v2_robust.py train <seed> ; python src/unified/marl_v2_robust.py report
Writes reports/unified/marl_v2_robust.json
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from unified import marl_v2 as m  # noqa: E402
from unified import marl_v2_shift as sh  # noqa: E402
from unified.agents import SharedMADDPG, TrainConfig  # noqa: E402
from unified.env import UnifiedSlicingEnv, run_policy  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CK = ROOT / "data" / "processed" / "unified"
EPISODES, VAL_EVERY, VAL_DAYS = 4000, 250, (41, 45)
SCEN = sh.SCENARIOS


class RandomMixEnv(UnifiedSlicingEnv):
    """Rescales the demand and forecast by a random per-slice factor at every reset."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.base_demand, self.base_forecast = self.demand.copy(), None if self.forecast is None else self.forecast.copy()
        self.mix_rng = np.random.default_rng(kw.get("seed", 0) + 1)

    def reset(self, start=None):
        g = self.mix_rng
        mult = np.array([g.uniform(1, 1.5) if g.random() < 0.5 else 1.0, np.exp(g.uniform(0, np.log(3))) if g.random() < 0.5 else 1.0,
                         np.exp(g.uniform(0, np.log(10))) if g.random() < 0.5 else 1.0], np.float32)
        self.demand = self.base_demand * mult
        if self.base_forecast is not None:
            self.forecast = self.base_forecast * mult
        return super().reset(start)


def week(env, pol, k=7):
    return [run_policy(env, pol, 144 * i + 1, m.DAY)["mean_reward"] for i in range(k)]


def scen_env(D, F, mult, a, b):
    k = np.array(mult, np.float32)
    return m.make_env(D * k, F * k, a, b)


def train(seed):
    torch.set_num_threads(3)
    D, F = m.build_data()
    a, b = m.TRAIN_DAYS[0] * m.DAY, m.TRAIN_DAYS[1] * m.DAY
    base = m.make_env(D, F, a, b, seed)
    env = RandomMixEnv(D[a:b], m.nbrs(), base.cfg, episode_len=36, start_range=(0, b - a - 41), forecast=F[a:b], seed=seed)
    vals = [scen_env(D, F, mult, VAL_DAYS[0] * m.DAY - 1, VAL_DAYS[1] * m.DAY) for mult in SCEN.values()]
    cfg = TrainConfig(episodes=EPISODES, seed=seed)
    agent = SharedMADDPG(env.N, env.obs_dim, env.nbr_matrix, cfg)
    best, best_state, curve, steps, t0 = -1.0, None, [], 0, time.time()
    for ep in range(EPISODES):
        sigma = cfg.noise_start + (cfg.noise_end - cfg.noise_start) * ep / (EPISODES - 1)
        obs, done = env.reset(), False
        while not done:
            act = agent.act(obs, sigma)
            obs2, r, team, done, _ = env.step(act)
            agent.replay.add(obs, act, team, r.mean(), obs2, float(done))
            obs = obs2
            steps += 1
            if steps > cfg.warmup_steps:
                agent.update()
        if (ep + 1) % VAL_EVERY == 0:
            v = float(np.mean([np.mean(week(e, agent.policy, 4)) for e in vals]))
            curve.append((ep + 1, round(v, 4)))
            if v > best:
                best, best_state = v, copy.deepcopy(agent.actor.state_dict())
            print(f"robust seed {seed} ep {ep + 1}: validation (mean over 3 mixes) {v:.4f} (best {best:.4f}) {time.time() - t0:.0f}s", flush=True)
    torch.save({"actor": best_state, "val": best, "curve": curve, "n": env.N, "f": env.obs_dim}, CK / f"marl_v2_robust_seed{seed}.pt")
    print("saved", seed)


def report():
    D, F = m.build_data()
    base, end = m.TEST_DAYS[0] * m.DAY - 1, m.TEST_DAYS[1] * m.DAY
    ex = json.loads((ROOT / "reports" / "unified" / "marl_v2_shift_experts.json").read_text(encoding="utf-8"))["results"]
    out = {"scenarios": {k: list(v) for k, v in SCEN.items()}, "seeds": {}}
    actors = []
    for f in sorted(CK.glob("marl_v2_robust_seed*.pt")):
        ck = torch.load(f, weights_only=False)
        test0 = scen_env(D, F, (1, 1, 1), base, end)
        a = SharedMADDPG(ck["n"], ck["f"], test0.nbr_matrix, TrainConfig())
        a.actor.load_state_dict(ck["actor"])
        actors.append(a)
        row = {}
        for name, mult in SCEN.items():
            per = week(scen_env(D, F, mult, base, end), a.policy)
            row[name] = {"test_per_day": [round(x, 4) for x in per], "test_mean": round(float(np.mean(per)), 4),
                         "tuned_expert_on_ordinary": round(float(np.mean(ex[name]["expert_forecast_tuned_on_baseline"])), 4),
                         "expert_retuned_on_this_mix": round(float(np.mean(ex[name]["expert_forecast_retuned"])), 4)}
        out["seeds"][f.stem.split("seed")[-1]] = {"validation": round(ck["val"], 4), **row}
        print(f.stem, {k: v["test_mean"] for k, v in row.items()}, flush=True)

    def ens(env, obs):
        with torch.no_grad():
            lg = torch.stack([x.actor(torch.from_numpy(obs)) for x in actors]).mean(0)
        return torch.softmax(lg, dim=-1).numpy()
    out["ensemble"] = {}
    for name, mult in SCEN.items():
        per = week(scen_env(D, F, mult, base, end), ens)
        ref = np.array(ex[name]["expert_forecast_retuned"])
        g = np.array(per) - ref
        out["ensemble"][name] = {"test_per_day": [round(x, 4) for x in per], "test_mean": round(float(np.mean(per)), 4), "gain_over_retuned_expert": round(float(g.mean()), 4),
                                 "ci95": round(float(2.447 * g.std(ddof=1) / 7 ** 0.5), 4), "days_positive": int((g > 0).sum())}
        print("ensemble", name, out["ensemble"][name], flush=True)
    (ROOT / "reports" / "unified" / "marl_v2_robust.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    train(int(sys.argv[2])) if sys.argv[1] == "train" else report()
