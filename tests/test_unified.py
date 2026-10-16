import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from unified.data import block_cells, grid_neighbours
from unified.env import ModelConfig, UnifiedSlicingEnv, headroom, qos


def _env(demand=None, forecast=None):
    cells = block_cells(2)  # 2x2 block, every cell has two neighbours
    if demand is None:
        demand = np.full((10, 4, 3), [40.0, 5.0, 2.0])
    return UnifiedSlicingEnv(demand, grid_neighbours(cells), ModelConfig(), episode_len=5,
                             start_range=(0, 3), forecast=forecast)


def test_block_and_neighbours_follow_the_real_grid():
    cells = block_cells(3)
    assert cells[0] + 1 == cells[1] and cells[0] + 100 == cells[3]
    nb = grid_neighbours(cells)
    assert sorted(nb[4]) == [1, 3, 5, 7]  # centre cell of a 3x3 block
    assert len(nb[0]) == 2                 # corner cell


def test_overload_loses_traffic_and_violates_sla():
    cfg = ModelConfig()
    q = qos(np.array([[50.0, 5.0, 2.0]]), np.array([[25.0, 20.0, 20.0]]), cfg)
    assert q["served"][0, 0] == pytest.approx(25.0)
    assert q["loss"][0, 0] == pytest.approx(0.5 + cfg.residual_loss[0])
    assert not q["sla_ok"][0, 0]
    assert q["sla_ok"][0, 1] and q["sla_ok"][0, 2]  # 25% and 10% utilisation: fine


def test_latency_rises_with_utilisation_and_slice_without_traffic_is_ok():
    cfg = ModelConfig()
    low = qos(np.array([[10.0, 0.0, 1.0]]), np.array([[50.0, 10.0, 10.0]]), cfg)
    high = qos(np.array([[45.0, 0.0, 1.0]]), np.array([[50.0, 10.0, 10.0]]), cfg)
    assert high["latency_p95_ms"][0, 0] > low["latency_p95_ms"][0, 0]
    assert low["sla_ok"][0, 1]  # URLLC has no demand


def test_neighbour_load_reduces_capacity():
    env = _env()
    env.reset(start=0)
    full = np.tile([0.4, 0.4, 0.2, 0.0], (4, 1))
    _, _, _, _, info_first = env.step(full)
    _, _, _, _, info_second = env.step(full)
    assert np.all(info_first["capacity_mbps"] == pytest.approx(100.0))
    assert np.all(info_second["capacity_mbps"] < 100.0)


def test_actions_are_normalised_and_rewards_bounded():
    env = _env()
    env.reset(start=0)
    _, reward, team, _, info = env.step(np.tile([2.0, 1.0, 1.0, 0.0], (4, 1)))
    assert np.allclose(env.prev_action.sum(axis=1), 1.0)
    assert np.all(reward <= 1.0) and np.all(reward >= -0.1)
    assert np.all(info["served"] <= info["demand_mbps"] + 1e-9)


def test_latency_variation_is_off_by_default_and_smooths_sla_when_on():
    d, a = np.full((4000, 3), [34.0, 0.0, 0.0]), np.full((4000, 3), [50.0, 10.0, 10.0])  # eMBB at 68% load
    fixed = qos(d, a, ModelConfig(), np.random.default_rng(0))
    assert len(np.unique(fixed["latency_p95_ms"][:, 0])) == 1
    varied = qos(d, a, ModelConfig(latency_sigma=0.21), np.random.default_rng(0))
    rate = varied["sla_ok"][:, 0].mean()
    assert 0.2 < rate < 0.9  # near the SLA edge some windows pass and some fail


def test_capacity_factor_models_an_outage():
    env = _env()
    env.capacity_factor = np.ones((10, 4))
    env.capacity_factor[1, 0] = 0.0
    env.reset(start=0)
    _, _, _, _, info = env.step(np.tile([0.4, 0.4, 0.2, 0.0], (4, 1)))
    assert info["served"][0].sum() == pytest.approx(0.0, abs=1e-6)
    assert not info["sla_ok"][0].any()


def test_forecast_adds_observation_features():
    plain = _env()
    with_fc = _env(forecast=np.ones((10, 4, 3)))
    assert with_fc.obs_dim == plain.obs_dim + 3
    assert with_fc.reset(start=0).shape == (4, with_fc.obs_dim)


def test_headroom_policy_keeps_urllc_under_its_safe_utilisation():
    env = _env()
    env.reset(start=0)
    shares = headroom(env, None, "current")
    assert np.allclose(shares.sum(axis=1), 1.0)
    urllc_capacity = shares[:, 1] * 100.0
    assert np.all(5.0 / urllc_capacity <= 0.5 + 1e-9)


# ---------- Digital Twin dataset scenarios (synthetic grid, no Milan files needed) ----------
def _provider(scenario, n_side=4, seed=0, events=None):
    import pandas as pd
    from unified.twin_dataset import PER_DAY, REAL_DAYS, DayProvider, build_calendar
    rng = np.random.default_rng(seed)
    n = n_side * n_side
    grid = rng.uniform(0.0, 1.0, (REAL_DAYS * PER_DAY, n, 3)).astype(np.float32)
    rows, cols = np.divmod(np.arange(n), n_side)
    tbl = pd.DataFrame({"industrial": np.arange(n) % 4 == 0, "grid_row": rows, "grid_col": cols})
    nbrs = [[r * n_side + c for r, c in ((i // n_side + dr, i % n_side + dc) for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)))
             if 0 <= r < n_side and 0 <= c < n_side] for i in range(n)]
    cal = build_calendar(7, seed)
    scale = np.array([10.0, 2.0, 1.0], np.float32)
    return DayProvider(scenario, grid, list(range(n)), cal, scale, events, nbrs, tbl, seed), cal


def test_scenarios_share_the_same_underlying_day():
    base, _ = _provider("baseline")
    iot, _ = _provider("massive_iot")
    b, m = base.get(4)["demand"], iot.get(4)["demand"]
    np.testing.assert_allclose(m[:, :, :2], b[:, :, :2])                 # eMBB / URLLC untouched
    on_hour = np.arange(144) % 6 == 0
    np.testing.assert_allclose(m[on_hour, :, 2], 30 * b[on_hour, :, 2], rtol=1e-5)
    np.testing.assert_allclose(m[~on_hour, :, 2], 10 * b[~on_hour, :, 2], rtol=1e-5)


def test_industrial_urllc_only_on_industrial_cells_and_weekday_shifts():
    base, cal = _provider("baseline")
    ind, _ = _provider("industrial_urllc")
    day = int(cal.index[(~cal.is_weekend) & (~cal.is_holiday)][0])
    b, d = base.get(day)["demand"][:, :, 1], ind.get(day)["demand"][:, :, 1]
    industrial = np.arange(16) % 4 == 0
    np.testing.assert_allclose(d[:, ~industrial], b[:, ~industrial])
    np.testing.assert_allclose(d[60, industrial], 4 * b[60, industrial], rtol=1e-5)   # 10:00, on shift
    np.testing.assert_allclose(d[10, industrial], 1.5 * b[10, industrial], rtol=1e-5)  # 01:40, off shift


def test_energy_saving_sleeps_quiet_cells_at_night_and_conserves_traffic():
    base, _ = _provider("baseline")
    es, _ = _provider("energy_saving")
    b, e = base.get(2), es.get(2)
    np.testing.assert_allclose(e["demand"].sum(axis=1), b["demand"].sum(axis=1), rtol=1e-5)  # nothing is lost
    asleep = e["asleep"]
    assert asleep.any() and not asleep[:6].any() and not asleep[30:].any()               # only 01:00-05:00
    assert (e["demand"][asleep] == 0).all() and (e["cap_f"][asleep] == 0).all()
    np.testing.assert_allclose(e["r_in"].sum(), e["r_out"].sum(), rtol=1e-5)


def test_cell_outage_moves_traffic_to_neighbours():
    import pandas as pd
    ev = pd.DataFrame([{"event_id": 0, "type": "cell_outage", "code": 2, "start_interval": 2 * 144 + 10,
                        "end_interval": 2 * 144 + 20, "cells_local": [5], "params": {"capacity_x": 0.0, "rehome_fraction": 0.7}}])
    base, _ = _provider("baseline")
    out, _ = _provider("events", events=ev)
    b, o = base.get(2), out.get(2)
    assert (o["cap_f"][10:20, 5] == 0).all() and (o["code"][10:20, 5] == 2).all() and o["code"][20, 5] == 0
    np.testing.assert_allclose(o["demand"][15, 5], 0.3 * b["demand"][15, 5], rtol=1e-5)
    np.testing.assert_allclose(o["demand"][15].sum(), b["demand"][15].sum(), rtol=1e-5)   # 70% re-homed, not lost
