"""Unified multi-cell slicing environment (and dataset generator).

One record per (interval, cell, slice) joins everything the separate datasets
could not: real traffic and real spatial layout (Milan), slice demand, an
allocation action, and the resulting QoS, SLA outcome and reward.

What is real and what is modelled
---------------------------------
Real:      per-cell traffic over time and the grid adjacency (Milan, see data.py).
Calibrated: latency vs. utilisation, from a published RAN telemetry dataset
            (calibrate.py -> qos_calibration.json).
Modelled:  slice mapping (data.py), demand scale, cell capacity, inter-cell
            interference, slice latency scales, residual loss, reward weights.
            All of these live in `ModelConfig` so they are visible and testable.

Dynamics for one step at interval t (decisions apply to the *next* interval, so
a forecast of t+1 is useful):
  demand  d[t+1, i, s]                      (Mbps)
  capacity C_i = C / (1 + coupling * mean neighbour load at t)   (load-coupled interference)
  allocated c_is = share_is * C_i, share has a 4th "idle" entry that leaves PRBs unused
  utilisation u = d / c;   served = min(d, c);   loss = overflow + residual
  latency p50/p95 = calibrated curve(u) * slice scale;  jitter = p95 - p50
  SLA ok = p95 <= target, jitter <= target, 1 - loss >= reliability, served >= 99% of demand
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SLICES = ("eMBB", "URLLC", "mMTC")
CALIBRATION = json.loads((Path(__file__).resolve().parent / "qos_calibration.json").read_text(encoding="utf-8"))


@dataclass
class ModelConfig:
    capacity_mbps: float = 100.0
    # 95th percentile of each slice's demand (over training cells and intervals) as a fraction of capacity.
    demand_p95_fraction: tuple = (0.80, 0.10, 0.05)
    coupling: float = 0.3                      # interference from neighbour load
    latency_scale: tuple = (1.0, 0.1, 2.0)     # URLLC scheduled with priority, mMTC tolerant
    residual_loss: tuple = (1e-4, 1e-6, 1e-3)
    # SLA targets: same values as marl/b5g_env.py QOS_TARGETS.
    target_latency_ms: tuple = (50.0, 5.0, 100.0)
    target_jitter_ms: tuple = (20.0, 1.0, 50.0)
    target_reliability: tuple = (0.99, 0.99999, 0.99)
    slice_weight: tuple = (0.35, 0.45, 0.20)
    w_sla: float = 0.7
    w_served: float = 0.3
    w_churn: float = 0.1
    team_mix: float = 0.5                      # team reward = (1-mix)*own + mix*neighbour mean
    # Per-window lognormal latency variation. 0 = expected-value model (used by the RL experiments);
    # the datasets use CALIBRATION["latency_sigma"], fitted so the SLA-vs-load curve matches the reference.
    latency_sigma: float = 0.0


def demand_scale(activity_train: np.ndarray, cfg: ModelConfig) -> np.ndarray:
    """Per-slice multiplier mapping Milan activity units to Mbps, fit on training intervals only."""
    p95 = np.percentile(activity_train.reshape(-1, 3), 95, axis=0)
    return np.array(cfg.demand_p95_fraction) * cfg.capacity_mbps / np.maximum(p95, 1e-9)


def _curve(u: np.ndarray, which: str) -> np.ndarray:
    xs = np.array(CALIBRATION["utilisation"])
    ys = np.array(CALIBRATION[which])
    inside = np.interp(np.clip(u, 0.0, 1.0), xs, ys)
    # Above full utilisation the queue keeps growing: latency rises linearly with the overload, capped at 3x.
    return inside * np.clip(1.0 + np.maximum(u - 1.0, 0.0), 1.0, 3.0)


def qos(demand: np.ndarray, allocated: np.ndarray, cfg: ModelConfig, rng: np.random.Generator | None = None) -> dict:
    """QoS and SLA for arrays shaped [..., 3]."""
    eps = 1e-9
    u = np.where(demand > 0, demand / np.maximum(allocated, eps), 0.0)
    served = np.minimum(demand, allocated)
    scale = np.array(cfg.latency_scale)
    if cfg.latency_sigma > 0 and rng is not None:
        scale = scale * np.exp(rng.normal(0.0, cfg.latency_sigma, size=u.shape))
    p50 = _curve(u, "lat_p50_ms") * scale
    p95 = _curve(u, "lat_p95_ms") * scale
    jitter = p95 - p50
    overflow = np.where(demand > 0, np.maximum(0.0, 1.0 - allocated / np.maximum(demand, eps)), 0.0)
    loss = np.minimum(1.0, overflow + np.array(cfg.residual_loss))
    ok = ((p95 <= np.array(cfg.target_latency_ms))
          & (jitter <= np.array(cfg.target_jitter_ms))
          & (1.0 - loss >= np.array(cfg.target_reliability))
          & (served >= 0.99 * demand))
    ok = np.where(demand > 0, ok, True)  # a slice with no traffic is not violated
    return {"utilisation": u, "served": served, "latency_p50_ms": p50, "latency_p95_ms": p95,
            "jitter_ms": jitter, "loss": loss, "sla_ok": ok}


class UnifiedSlicingEnv:
    """Vectorised over cells. Actions are [N, 4] shares (eMBB, URLLC, mMTC, idle) summing to 1."""

    ACTION_DIM = 4

    def __init__(self, demand: np.ndarray, neighbours: list[list[int]], cfg: ModelConfig | None = None,
                 episode_len: int = 36, start_range: tuple[int, int] | None = None,
                 forecast: np.ndarray | None = None, seed: int = 0,
                 capacity_factor: np.ndarray | None = None, coupling_factor: np.ndarray | None = None):
        # [T, N, 3] Mbps; float32 is kept as is so year-long runs fit in memory.
        self.demand = demand if demand.dtype in (np.float32, np.float64) else demand.astype(np.float64)
        # Optional per-interval, per-cell multipliers for events (outage, degradation, interference spikes).
        self.capacity_factor = capacity_factor
        self.coupling_factor = coupling_factor
        self.T, self.N, _ = demand.shape
        self.neighbours = neighbours
        self.cfg = cfg or ModelConfig()
        self.episode_len = episode_len
        self.start_range = start_range or (0, self.T - episode_len - 1)
        self.forecast = forecast                            # [T, N, 3] Mbps forecast of d[t+1] made at t, or None
        self.rng = np.random.default_rng(seed)
        rows = [i for i, nb in enumerate(neighbours) for _ in nb]
        cols = [j for nb in neighbours for j in nb]
        vals = [1.0 / len(nb) for nb in neighbours for _ in nb]
        if self.N > 2000:  # city-scale grids: a dense N x N matrix would not fit in memory
            from scipy.sparse import csr_matrix
            self.nbr_matrix = csr_matrix((vals, (rows, cols)), shape=(self.N, self.N))
        else:
            self.nbr_matrix = np.zeros((self.N, self.N))
            self.nbr_matrix[rows, cols] = vals
        self.obs_dim = 14 + (3 if forecast is not None else 0)

    # ---------- episode control ----------
    def reset(self, start: int | None = None) -> np.ndarray:
        lo, hi = self.start_range
        self.t = int(self.rng.integers(lo, hi + 1)) if start is None else int(start)
        self.t_end = min(self.t + self.episode_len, self.T - 1)
        self.prev_action = np.tile([1 / 3, 1 / 3, 1 / 3, 0.0], (self.N, 1))
        self.prev_ok = np.ones((self.N, 3))
        self.load = np.zeros(self.N)
        return self._obs()

    def _obs(self) -> np.ndarray:
        C = self.cfg.capacity_mbps
        minute = (self.t % 144) / 144.0
        parts = [self.demand[self.t] / C, self.prev_action, self.prev_ok,
                 self.load[:, None], (self.nbr_matrix @ self.load)[:, None],
                 np.tile([np.sin(2 * np.pi * minute), np.cos(2 * np.pi * minute)], (self.N, 1))]
        if self.forecast is not None:
            parts.append(self.forecast[self.t] / C)
        return np.concatenate(parts, axis=1).astype(np.float32)

    def capacity_at(self, t: int) -> np.ndarray:
        """Effective capacity per cell at interval t given the current neighbour load."""
        cf = 1.0 if self.capacity_factor is None else self.capacity_factor[t]
        kf = 1.0 if self.coupling_factor is None else self.coupling_factor[t]
        return np.maximum(self.cfg.capacity_mbps * cf / (1.0 + self.cfg.coupling * kf * (self.nbr_matrix @ self.load)), 1e-6)

    def step(self, action: np.ndarray):
        cfg = self.cfg
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, 0.0, None)
        action = action / np.maximum(action.sum(axis=1, keepdims=True), 1e-9)
        d = self.demand[self.t + 1].astype(np.float64)
        cap = self.capacity_at(self.t + 1)
        allocated = action[:, :3] * cap[:, None]
        q = qos(d, allocated, cfg, self.rng)

        w = np.array(cfg.slice_weight)
        sla_score = (q["sla_ok"] * w).sum(axis=1)
        total = d.sum(axis=1)
        served_frac = np.where(total > 0, q["served"].sum(axis=1) / np.maximum(total, 1e-9), 1.0)
        churn = 0.5 * np.abs(action - self.prev_action).sum(axis=1)
        reward = cfg.w_sla * sla_score + cfg.w_served * served_frac - cfg.w_churn * churn
        team = (1 - cfg.team_mix) * reward + cfg.team_mix * (self.nbr_matrix @ reward)

        info = {k: v for k, v in q.items()}
        info.update({"capacity_mbps": cap, "demand_mbps": d, "allocated_mbps": allocated,
                     "reward": reward, "sla_score": sla_score, "served_frac": served_frac, "churn": churn})
        self.load = np.clip(q["served"].sum(axis=1) / cap, 0.0, 1.0)
        self.prev_action = action
        self.prev_ok = q["sla_ok"].astype(np.float64)
        self.t += 1
        done = self.t >= self.t_end
        return self._obs(), reward, team, done, info


# ---------- reference policies ----------
def equal_split(env: UnifiedSlicingEnv, obs: np.ndarray) -> np.ndarray:
    return np.tile([1 / 3, 1 / 3, 1 / 3, 0.0], (env.N, 1))


def proportional(env: UnifiedSlicingEnv, obs: np.ndarray, source: str = "current") -> np.ndarray:
    """Share capacity in proportion to demand: the 'traffic-aware' baseline common in the literature.
    source: 'current' (persistence), 'forecast' (env.forecast) or 'oracle' (true next demand)."""
    d = np.maximum(_demand_source(env, source), 1e-6)
    shares = d / d.sum(axis=1, keepdims=True)
    return np.concatenate([shares, np.zeros((env.N, 1))], axis=1)


def _demand_source(env: UnifiedSlicingEnv, source: str) -> np.ndarray:
    if source == "current":
        return env.demand[env.t]
    if source == "forecast":
        return env.forecast[env.t]
    return env.demand[env.t + 1]


# Highest utilisation at which the calibrated curve still meets each slice's latency and jitter target.
HEADROOM_UTILISATION = (0.65, 0.50, 0.65)


def headroom(env: UnifiedSlicingEnv, obs: np.ndarray, source: str = "current") -> np.ndarray:
    """Expert baseline: give URLLC and mMTC enough capacity to stay under their SLA-safe
    utilisation, and the remainder to eMBB (priority + headroom, as an operator would)."""
    d = _demand_source(env, source)
    cap = env.capacity_at(env.t)  # what the controller can observe when it decides
    need = d / np.array(HEADROOM_UTILISATION) / cap[:, None]
    urllc = np.minimum(need[:, 1], 1.0)
    mmtc = np.minimum(need[:, 2], 1.0 - urllc)
    embb = 1.0 - urllc - mmtc
    return np.stack([embb, urllc, mmtc, np.zeros(env.N)], axis=1)


def run_policy(env: UnifiedSlicingEnv, policy, start: int, steps: int) -> dict:
    """Roll a policy over a fixed window; returns per-step system metrics."""
    obs = env.reset(start=start)
    env.t_end = min(start + steps, env.T - 1)
    rewards, sla, slice_ok, served = [], [], [], []
    done = False
    while not done:
        obs, r, _, done, info = env.step(policy(env, obs))
        rewards.append(r.mean())
        sla.append(info["sla_score"].mean())
        slice_ok.append(info["sla_ok"].mean(axis=0))
        served.append(info["served_frac"].mean())
    return {"mean_reward": float(np.mean(rewards)), "sla_score": float(np.mean(sla)),
            "slice_sla_ok": dict(zip(SLICES, np.mean(slice_ok, axis=0).round(4).tolist())),
            "served_frac": float(np.mean(served)), "steps": len(rewards)}
