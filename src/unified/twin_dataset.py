"""Unified Digital Twin dataset generator, v2: city-scale, multi-scenario, ~120 columns.

Two standard builds (see docs/Unified_Twin_Dataset.md):
  core year : --block 20  --days 364   400 cells, 1 Nov 2013 - 30 Oct 2014
  full city : --block 100 --days 62    all 10,000 Milan cells, the 62 recorded days

Backbone (real): Telecom Italia Milan traffic on the real 100 x 100 grid, real cell
centroids and adjacency, Milan's real holidays. Days after the 62 recorded ones are
bootstrapped from same-weekday real days with a growth trend (`source` labels every row).

Scenarios (every one except baseline also carries injected events):
  baseline          recorded traffic, no events
  events            + flash crowds, outages, interference, URLLC bursts, mMTC storms, degradations
  growth_6g         + eMBB x1.3, URLLC x2, mMTC x3
  energy_saving     + night-time cell sleep (01:00-05:00, checkerboard, idle cells), traffic handed to neighbours
  industrial_urllc  + an industrial zone (12% of cells) with URLLC x4 on weekday shifts, x1.5 otherwise
  massive_iot       + mMTC x10 with synchronised on-the-hour reporting (x3 in the first interval of each hour)

Outcomes come from env.py (calibrated latency/jitter, loss, SLA, load-coupled
interference, reward). The generator streams one day at a time from a disk-backed
copy of the real traffic, so the full city runs in a few GB of memory.

Run: python src/unified/twin_dataset.py --block 20 --days 364
     python src/unified/twin_dataset.py --block 100 --days 62 --scenarios baseline,events
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from milan_features import aggregate_one_day  # noqa: E402
from unified.data import GRID, block_cells, grid_neighbours  # noqa: E402
from unified.env import (CALIBRATION, SLICES, ModelConfig, UnifiedSlicingEnv,  # noqa: E402
                         equal_split, headroom, proportional)

MILAN = r"E:\FYP DATA\6G\data\raw\milan_telecom_italia_full\data"
BASE = Path(r"E:\FYP DATA\6G\data\processed\unified_twin")
COMMIT_ROOT = Path(__file__).resolve().parents[2] / "reports" / "unified_twin"
START = dt.date(2013, 11, 1)
REAL_DAYS, PER_DAY = 62, 144
DUOMO = (45.4641, 9.1919)  # Milan cathedral, the city centre

HOLIDAYS = {
    dt.date(2013, 11, 1): "Ognissanti", dt.date(2013, 12, 7): "Sant'Ambrogio (Milan)",
    dt.date(2013, 12, 8): "Immacolata", dt.date(2013, 12, 25): "Natale", dt.date(2013, 12, 26): "Santo Stefano",
    dt.date(2014, 1, 1): "Capodanno", dt.date(2014, 1, 6): "Epifania", dt.date(2014, 4, 20): "Pasqua",
    dt.date(2014, 4, 21): "Pasquetta", dt.date(2014, 4, 25): "Liberazione", dt.date(2014, 5, 1): "Festa del Lavoro",
    dt.date(2014, 6, 2): "Festa della Repubblica", dt.date(2014, 8, 15): "Ferragosto",
    dt.date(2014, 11, 1): "Ognissanti", dt.date(2014, 12, 7): "Sant'Ambrogio (Milan)", dt.date(2014, 12, 8): "Immacolata",
}
GROWTH_PER_YEAR = 0.20
SCENARIOS = ("baseline", "events", "growth_6g", "energy_saving", "industrial_urllc", "massive_iot")
USER_RATE_MBPS = np.array([4.0, 0.2, 0.02], np.float32)     # assumed mean rate per active user
PACKET_BYTES = np.array([1500, 32, 125], np.float32)       # eMBB MTU; URLLC 32 B (3GPP TR 38.913); mMTC small report
PRBS = 106                                                  # 20 MHz NR carrier, 15 kHz SCS
EARTH = dict(n_trx=6, p0=130.0, dp=4.7, pmax=20.0, psleep=75.0)   # EARTH macro BS power model (Auer et al., 2011)
CQI_SINR_DB = [-6.7, -4.7, -2.3, 0.2, 2.4, 4.3, 5.9, 8.1, 10.3, 11.7, 14.1, 16.3, 18.7, 21.0, 22.7]
CQI_EFFICIENCY = np.array([0.0, 0.1523, 0.2344, 0.3770, 0.6016, 0.8770, 1.1758, 1.4766, 1.9141,
                           2.4063, 2.7305, 3.3223, 3.9023, 4.5234, 5.1152, 5.5547], np.float32)  # 3GPP TS 38.214 Table 5.2.2.1-2
MODULATION = np.array(["none"] + ["QPSK"] * 6 + ["16QAM"] * 3 + ["64QAM"] * 6)
POLICIES = ("headroom", "headroom_explore", "proportional", "equal_split", "random")
POLICY_P = (0.45, 0.25, 0.12, 0.08, 0.10)
EVENTS = {  # type: (code, rate per day per 400 cells, duration range in intervals)
    "flash_crowd": (1, 0.5, (6, 18)), "cell_outage": (2, 0.2, (3, 24)),
    "interference_spike": (3, 0.3, (6, 36)), "urllc_burst": (4, 0.3, (2, 6)),
    "mmtc_storm": (5, 0.3, (1, 3)), "capacity_degradation": (6, 0.1, (12, 72)),
}
CELL_SLEEP = 7
CODE_NAMES = ["none"] + list(EVENTS) + ["cell_sleep"]


# ---------------------------------------------------------------- real traffic, disk-backed
def full_grid(path: Path) -> np.memmap:
    """[62*144, 10000, 3] float32 activity (internet, call, sms) for the whole grid, built once."""
    if path.exists():
        return np.load(path, mmap_mode="r")
    files = sorted(Path(MILAN).glob("sms-call-internet-mi-*.txt"))[:REAL_DAYS]
    arr = np.lib.format.open_memmap(path.with_suffix(".tmp.npy"), mode="w+", dtype=np.float32,
                                    shape=(REAL_DAYS * PER_DAY, GRID * GRID, 3))
    for k, f in enumerate(files):
        t0 = time.time()
        day = aggregate_one_day(str(f))
        midnight = dt.datetime.combine(START + dt.timedelta(k), dt.time()) - dt.timedelta(hours=1)  # CET -> UTC
        ms0 = int(midnight.replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
        iod = ((day["time_interval_ms"].to_numpy() - ms0) // 600_000).astype(int)
        keep = (iod >= 0) & (iod < PER_DAY)
        cell = day["square_id"].to_numpy()[keep] - 1
        block = np.zeros((PER_DAY, GRID * GRID, 3), np.float32)
        block[iod[keep], cell, 0] = day["internet_traffic"].to_numpy()[keep]
        block[iod[keep], cell, 1] = (day["call_in"] + day["call_out"]).to_numpy()[keep]
        block[iod[keep], cell, 2] = (day["sms_in"] + day["sms_out"]).to_numpy()[keep]
        arr[k * PER_DAY:(k + 1) * PER_DAY] = block
        print(f"  grid cache day {k + 1}/{REAL_DAYS} ({time.time() - t0:.1f}s)", flush=True)
    arr.flush()
    del arr
    path.with_suffix(".tmp.npy").replace(path)
    return np.load(path, mmap_mode="r")


# ---------------------------------------------------------------- calendar
def build_calendar(days: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    real = [START + dt.timedelta(d) for d in range(REAL_DAYS)]
    holiday_pool = [i for i, d in enumerate(real) if d in HOLIDAYS]
    regular = {w: [i for i, d in enumerate(real) if d.weekday() == w and d not in HOLIDAYS
                   and not (dt.date(2013, 12, 21) <= d <= dt.date(2014, 1, 1))] for w in range(7)}
    seasons = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
               6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}
    rows = []
    for k in range(days):
        date = START + dt.timedelta(k)
        if k < REAL_DAYS:
            src, source, growth = k, "milan_real", 1.0
        else:
            pool = holiday_pool if date in HOLIDAYS else regular[date.weekday()]
            src, source, growth = int(rng.choice(pool)), "bootstrap", 1.0 + GROWTH_PER_YEAR * k / 365.0
        rows.append({"day_index": k, "date": date.isoformat(), "weekday": date.strftime("%a"), "month": date.month,
                     "day_of_year": date.timetuple().tm_yday, "iso_week": date.isocalendar()[1], "season": seasons[date.month],
                     "is_weekend": date.weekday() >= 5, "is_holiday": date in HOLIDAYS, "holiday_name": HOLIDAYS.get(date, ""),
                     "source": source, "source_date": real[src].isoformat(), "source_day_index": src,
                     "growth_factor": round(growth, 4)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- topology & static cell attributes
def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def cell_table(cells: list[int], nbrs, seed: int) -> pd.DataFrame:
    gj = json.loads(Path(MILAN, "milano-grid.geojson").read_text(encoding="utf-8"))
    cent = {}
    wanted = set(cells)
    for f in gj["features"]:
        cid = f["properties"]["cellId"]
        if cid in wanted:
            ring = np.array(f["geometry"]["coordinates"][0][:-1])
            cent[cid] = (float(ring[:, 1].mean()), float(ring[:, 0].mean()))
    lat = np.array([cent[c][0] for c in cells]); lon = np.array([cent[c][1] for c in cells])
    dist = haversine_km(lat, lon, *DUOMO)
    zone = np.where(dist < 2.0, "centre", np.where(dist < 5.0, "inner", "outer"))
    rng = np.random.default_rng(seed + 11)
    # Industrial zone (assumption for the industrial_urllc scenario): 12% of cells, three times
    # likelier in the outer ring, where Milan's industrial areas are.
    w = np.where(zone == "outer", 3.0, 1.0); w /= w.sum()
    industrial = np.zeros(len(cells), bool)
    industrial[rng.choice(len(cells), size=max(1, int(0.12 * len(cells))), replace=False, p=w)] = True
    base_sinr = np.clip(np.random.default_rng(seed).normal(14.4, 5.1, len(cells)), -5, 30)
    return pd.DataFrame({"cell_id": cells, "grid_row": [(c - 1) // GRID for c in cells], "grid_col": [(c - 1) % GRID for c in cells],
                         "lat": lat.round(6), "lon": lon.round(6), "distance_to_duomo_km": dist.round(3), "zone": zone,
                         "industrial": industrial, "neighbours": [";".join(str(cells[j]) for j in nb) for nb in nbrs],
                         "base_sinr_db": base_sinr.round(2), "nominal_capacity_mbps": 100.0, "prbs": PRBS})


# ---------------------------------------------------------------- events (shared with showcase.py)
def cluster(centre: int, radius: int, n_side: int) -> list[int]:
    r0, c0 = divmod(centre, n_side)
    return [r * n_side + c for r in range(max(0, r0 - radius), min(n_side, r0 + radius + 1))
            for c in range(max(0, c0 - radius), min(n_side, c0 + radius + 1)) if abs(r - r0) + abs(c - c0) <= radius]


def schedule_events(days: int, n_side: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 2)
    scale = n_side * n_side / 400.0
    rows, eid = [], 0
    for day in range(days):
        for etype, (code, rate, (dmin, dmax)) in EVENTS.items():
            for _ in range(rng.poisson(rate * scale)):
                centre = int(rng.integers(0, n_side * n_side))
                if etype == "flash_crowd":
                    start_iod = int(rng.integers(17 * 6, 22 * 6))
                    cells, params = cluster(centre, int(rng.integers(1, 3)), n_side), {
                        "embb_x": round(float(rng.uniform(2, 4)), 2), "urllc_x": round(float(rng.uniform(1.5, 2)), 2), "mmtc_x": 1.2}
                elif etype == "cell_outage":
                    start_iod, cells, params = int(rng.integers(0, PER_DAY)), [centre], {"capacity_x": 0.0, "rehome_fraction": 0.7}
                elif etype == "interference_spike":
                    start_iod, cells, params = int(rng.integers(0, PER_DAY)), cluster(centre, 1, n_side), {"coupling_x": 3.0}
                elif etype == "urllc_burst":
                    start_iod = int(rng.integers(0, PER_DAY))
                    cells = sorted(set(int(c) for c in rng.choice(cluster(centre, 1, n_side), size=int(rng.integers(1, 4)))))
                    params = {"urllc_x": round(float(rng.uniform(3, 6)), 2)}
                elif etype == "mmtc_storm":
                    start_iod, cells, params = int(rng.integers(0, PER_DAY)), cluster(centre, 2, n_side), {"mmtc_x": round(float(rng.uniform(5, 10)), 2)}
                else:
                    start_iod, cells, params = int(rng.integers(0, PER_DAY)), [centre], {"capacity_x": 0.5}
                start = day * PER_DAY + start_iod
                end = min(start + int(rng.integers(dmin, dmax + 1)), days * PER_DAY)
                rows.append({"event_id": eid, "type": etype, "code": code, "start_interval": start, "end_interval": end,
                             "cells_local": cells, "params": params})
                eid += 1
    return pd.DataFrame(rows)


def _apply_event(ev, demand, cap_f, coup_f, code, eid, r_in, r_out, nbrs, a, b, offset=0):
    """Apply one event to arrays covering global intervals [offset, offset + len)."""
    sa, sb = a - offset, b - offset
    cells, p = ev.cells_local, ev.params
    if ev.type in ("flash_crowd", "urllc_burst", "mmtc_storm"):
        demand[sa:sb, cells, :] *= np.array([p.get("embb_x", 1.0), p.get("urllc_x", 1.0), p.get("mmtc_x", 1.0)], np.float32)
    elif ev.type == "cell_outage":
        i = cells[0]
        cap_f[sa:sb, i] = 0.0
        moved = demand[sa:sb, i, :] * p["rehome_fraction"]
        demand[sa:sb, i, :] -= moved
        r_out[sa:sb, i] += moved.sum(axis=1)
        for j in nbrs[i]:
            demand[sa:sb, j, :] += moved / len(nbrs[i])
            r_in[sa:sb, j] += moved.sum(axis=1) / len(nbrs[i])
    elif ev.type == "interference_spike":
        coup_f[sa:sb, cells] *= p["coupling_x"]
    else:
        cap_f[sa:sb, cells] *= p["capacity_x"]
    code[sa:sb, cells] = ev.code
    eid[sa:sb, cells] = ev.event_id


def apply_events(demand: np.ndarray, events: pd.DataFrame, nbrs: list[list[int]]):
    """Whole-array version (used by showcase.py)."""
    T, n, _ = demand.shape
    cap_f, coup_f = np.ones((T, n), np.float32), np.ones((T, n), np.float32)
    code, eid = np.zeros((T, n), np.int8), np.full((T, n), -1, np.int32)
    r_in, r_out = np.zeros((T, n), np.float32), np.zeros((T, n), np.float32)
    for ev in events.itertuples():
        _apply_event(ev, demand, cap_f, coup_f, code, eid, r_in, r_out, nbrs, ev.start_interval, ev.end_interval)
    return cap_f, coup_f, code, eid, r_in, r_out


# ---------------------------------------------------------------- per-day scenario construction
class DayProvider:
    """Builds every per-day array for one scenario on demand, deterministically, with a small cache."""

    def __init__(self, scenario, grid, idx, calendar, scale, events, nbrs, cells_tbl, seed):
        self.s, self.grid, self.idx, self.cal, self.scale = scenario, grid, np.array(idx), calendar, scale
        self.events, self.nbrs, self.seed = events, nbrs, seed
        self.n = len(idx)
        self.industrial = cells_tbl["industrial"].to_numpy()
        rr, cc = cells_tbl["grid_row"].to_numpy(), cells_tbl["grid_col"].to_numpy()
        self.checker = ((rr + cc) % 2 == 0)
        self.by_day = {}
        if events is not None:
            for ev in events.itertuples():
                for d in range(ev.start_interval // PER_DAY, (ev.end_interval - 1) // PER_DAY + 1):
                    self.by_day.setdefault(d, []).append(ev)
        self.cache = {}

    def get(self, k: int) -> dict:
        if k not in self.cache:
            if len(self.cache) >= 3:
                self.cache.pop(min(self.cache))
            self.cache[k] = self._build(k)
        return self.cache[k]

    def _build(self, k: int) -> dict:
        row = self.cal.iloc[k]
        src = row.source_day_index
        act = np.array(self.grid[src * PER_DAY:(src + 1) * PER_DAY][:, self.idx, :], dtype=np.float32)
        rng = np.random.default_rng([self.seed, k])  # same bootstrap noise in every scenario
        if row.source == "bootstrap":
            act *= row.growth_factor * np.exp(rng.normal(0, 0.05, (1, self.n, 1))).astype(np.float32)
            act *= np.exp(rng.normal(0, 0.03, act.shape)).astype(np.float32)
        d = act * self.scale
        iod = np.arange(PER_DAY)
        if self.s == "growth_6g":
            d *= np.array([1.3, 2.0, 3.0], np.float32)
        elif self.s == "massive_iot":
            d[:, :, 2] *= 10.0
            d[iod % 6 == 0, :, 2] *= 3.0
        elif self.s == "industrial_urllc":
            shift = (iod >= 36) & (iod < 132) & (not row.is_weekend) & (not row.is_holiday)
            mult = np.where(shift[:, None], 4.0, 1.5) * self.industrial[None, :] + (~self.industrial[None, :])
            d[:, :, 1] *= mult.astype(np.float32)
        cap_f = np.ones((PER_DAY, self.n), np.float32)
        coup_f = np.ones((PER_DAY, self.n), np.float32)
        code = np.zeros((PER_DAY, self.n), np.int8)
        eid = np.full((PER_DAY, self.n), -1, np.int32)
        r_in = np.zeros((PER_DAY, self.n), np.float32)
        r_out = np.zeros((PER_DAY, self.n), np.float32)
        asleep = np.zeros((PER_DAY, self.n), bool)
        lo = k * PER_DAY
        for ev in self.by_day.get(k, []):
            a, b = max(ev.start_interval, lo), min(ev.end_interval, lo + PER_DAY)
            _apply_event(ev, d, cap_f, coup_f, code, eid, r_in, r_out, self.nbrs, a, b, offset=lo)
        if self.s == "energy_saving":
            # 01:00-05:00: lightly loaded cells on one colour of a checkerboard sleep, so every
            # sleeping cell keeps awake neighbours; all of its traffic is handed over to them.
            for t in range(6, 30):
                cand = self.checker & (d[t].sum(axis=1) < 15.0) & (code[t] == 0)
                for i in np.flatnonzero(cand):
                    moved = d[t, i].copy()
                    d[t, i] = 0.0
                    nb = self.nbrs[i]
                    if nb:
                        d[t, nb] += moved / len(nb)
                        r_in[t, nb] += moved.sum() / len(nb)
                        r_out[t, i] += moved.sum()
                cap_f[t, cand] = 0.0
                code[t, cand] = CELL_SLEEP
                asleep[t, cand] = True
        return {"demand": d, "cap_f": cap_f, "coup_f": coup_f, "code": code, "eid": eid,
                "r_in": r_in, "r_out": r_out, "asleep": asleep}


class View:
    """Array-like over the provider so the environment can index [t] across days lazily."""

    def __init__(self, provider: DayProvider, key: str, shape: tuple):
        self.p, self.key, self.shape, self.dtype = provider, key, shape, np.float32

    def __getitem__(self, t):
        return self.p.get(int(t) // PER_DAY)[self.key][int(t) % PER_DAY]


# ---------------------------------------------------------------- derived fields
def sinr_from_capacity(base_sinr_db, cap_fraction):
    eta0 = np.log2(1 + 10 ** (base_sinr_db / 10))
    sinr_lin = 2 ** (cap_fraction * eta0) - 1
    return np.clip(10 * np.log10(np.maximum(sinr_lin, 1e-3)), -10, 40)


def cqi_from_sinr(sinr_db):
    return np.clip(np.searchsorted(CQI_SINR_DB, sinr_db, side="right"), 0, 15).astype(np.int8)


def power_w(load, outage, asleep):
    e = EARTH
    p = np.where(load > 0, e["n_trx"] * (e["p0"] + e["dp"] * e["pmax"] * load), e["n_trx"] * e["psleep"])
    p = np.where(asleep, e["n_trx"] * e["psleep"], p)
    return np.where(outage, 0.0, p)


# ---------------------------------------------------------------- generation
def run_scenario(name, provider, cells, nbrs, cells_tbl, calendar, T, seed, out_dir: Path, per_file: str) -> dict:
    n = len(cells)
    cfg = ModelConfig(latency_sigma=CALIBRATION["latency_sigma"])
    demand_v = View(provider, "demand", (T, n, 3))
    env = UnifiedSlicingEnv(demand_v, nbrs, cfg, episode_len=T, seed=seed,
                            capacity_factor=View(provider, "cap_f", (T, n)), coupling_factor=View(provider, "coup_f", (T, n)))
    rng = np.random.default_rng([seed, 3])  # same policies and noise in every scenario, so scenarios differ only by design
    env.reset(start=0)
    env.t_end = T - 1
    base_sinr = cells_tbl["base_sinr_db"].to_numpy()
    policy_by_day = rng.choice(len(POLICIES), size=(T // PER_DAY + 2, n), p=POLICY_P)
    steps_per_flush = int(max(1, min(PER_DAY, 250_000 // n)))
    hist = []                                             # last 6 demand arrays (1 h rolling mean)
    ew_mean = ew_var = prev_d = None                    # EWMA of total demand, for the anomaly score
    writer = {"key": None, "w": None}
    buf, done = [], False
    stats = {"rows": 0, "sla": np.zeros(3), "reward": 0.0, "energy_kwh": 0.0, "served_tb": 0.0, "asleep": 0,
             "by_code_sla": np.zeros((len(CODE_NAMES), 3)), "by_code_reward": np.zeros(len(CODE_NAMES)),
             "by_code_rows": np.zeros(len(CODE_NAMES))}
    t_start = time.time()
    while not done:
        t = env.t
        h = headroom(env, None, "current")
        g = rng.gamma(np.maximum(h * 20, 0.05) + 0.05)
        cand = np.stack([h, 0.7 * h + 0.3 * g / g.sum(axis=1, keepdims=True), proportional(env, None, "current"),
                         equal_split(env, None), rng.dirichlet([1.0, 1.0, 1.0, 0.3], size=n)])
        pol = policy_by_day[(t + 1) // PER_DAY]
        action = cand[pol, np.arange(n)]
        nbr_load = env.nbr_matrix @ env.load
        _, reward, team, done, info = env.step(action)
        k = t + 1
        day = provider.get(k // PER_DAY)
        d = info["demand_mbps"].astype(np.float32)
        hist.append(d); hist = hist[-6:]
        total = d.sum(axis=1)
        if ew_mean is None:
            ew_mean, ew_var = total.copy(), (0.1 * total) ** 2 + 1.0
        z = (total - ew_mean) / np.sqrt(ew_var + 1e-6)
        ew_mean = 0.9 * ew_mean + 0.1 * total; ew_var = 0.9 * ew_var + 0.1 * (total - ew_mean) ** 2
        rec = {"k": k, "action": action, "nbr_load": nbr_load, "reward": reward, "team": team, "pol": pol, "info": info,
               "d": d, "d1h": np.mean(hist, axis=0), "delta": d - (prev_d if prev_d is not None else d), "z": z,
               "f1": demand_v[min(k + 1, T - 1)], "f3": demand_v[min(k + 3, T - 1)], "f6": demand_v[min(k + 6, T - 1)],
               "code": day["code"][k % PER_DAY], "eid": day["eid"][k % PER_DAY], "r_in": day["r_in"][k % PER_DAY],
               "r_out": day["r_out"][k % PER_DAY], "asleep": day["asleep"][k % PER_DAY],
               "cf": day["cap_f"][k % PER_DAY], "kf": day["coup_f"][k % PER_DAY]}
        prev_d = d
        buf.append(rec)
        c = rec["code"].astype(int)
        np.add.at(stats["by_code_sla"], c, info["sla_ok"]); np.add.at(stats["by_code_reward"], c, reward); np.add.at(stats["by_code_rows"], c, 1)
        if len(buf) >= steps_per_flush or (k + 1) % PER_DAY == 0 or done:
            flush(name, buf, cells, cells_tbl, calendar, base_sinr, env, rng, out_dir, writer, per_file, stats)
            buf = []
            day_no = k // PER_DAY
            if (k + 1) % PER_DAY == 0 and (day_no + 1) % 14 == 0:
                print(f"  {name}: {day_no + 1} days ({time.time() - t_start:.0f}s, {stats['rows']:,} rows)", flush=True)
    if writer["w"] is not None:
        writer["w"].close()
    rows = stats["rows"]
    by_event = {CODE_NAMES[c]: {"rows": int(stats["by_code_rows"][c]),
                                "sla_ok_rate": dict(zip(SLICES, (stats["by_code_sla"][c] / stats["by_code_rows"][c]).round(4).tolist())),
                                "mean_reward": round(float(stats["by_code_reward"][c] / stats["by_code_rows"][c]), 4)}
                for c in range(len(CODE_NAMES)) if stats["by_code_rows"][c] > 0}
    return {"rows": rows, "sla_ok_rate": dict(zip(SLICES, (stats["sla"] / rows).round(4).tolist())),
            "mean_reward": round(stats["reward"] / rows, 4), "energy_mwh": round(stats["energy_kwh"] / 1000, 2),
            "served_tb": round(stats["served_tb"], 2), "sleeping_cell_intervals": int(stats["asleep"]),
            "seconds": round(time.time() - t_start, 1), "by_event_type": by_event}


def flush(name, buf, cells, cells_tbl, calendar, base_sinr, env, rng, out_dir, writer, per_file, stats):
    n, steps = len(cells), len(buf)
    cfg = env.cfg
    st = lambda key: np.stack([b[key] for b in buf])
    inf = lambda key: np.stack([b["info"][key] for b in buf])
    flat = lambda a: a.reshape(steps * n, *a.shape[2:])
    rep = lambda a: np.repeat(np.asarray(a), n)
    k = np.array([b["k"] for b in buf])
    day_index = k // PER_DAY
    cal = calendar.set_index("day_index").loc[day_index]
    iod = k % PER_DAY
    action, cap = st("action"), inf("capacity_mbps")
    served, alloc = inf("served"), inf("allocated_mbps")
    d = st("d")
    p50, p95, jit, loss, ok = inf("latency_p50_ms"), inf("latency_p95_ms"), inf("jitter_ms"), inf("loss"), inf("sla_ok")
    load = np.clip(served.sum(axis=2) / cap, 0, 1)
    asleep, cf = st("asleep"), st("cf")
    outage = (cf == 0) & ~asleep
    sinr = sinr_from_capacity(base_sinr[None, :], cap / cfg.capacity_mbps)
    cqi = cqi_from_sinr(sinr)
    pw = power_w(load, outage, asleep)
    users = d / USER_RATE_MBPS
    tot_served = served.sum(axis=2)
    all_ok = ok.all(axis=2).astype(np.float32)
    nbr = env.nbr_matrix
    nbr_dem = np.stack([nbr @ x for x in d.sum(axis=2)])
    nbr_ok = np.stack([nbr @ x for x in all_ok])
    business = (iod >= 48) & (iod < 114) & ~cal["is_weekend"].to_numpy() & ~cal["is_holiday"].to_numpy()
    r3 = lambda a: np.round(a.astype(np.float32), 3)
    cols = {
        "scenario": name, "interval": rep(k), "day_index": rep(day_index), "date": rep(cal["date"].to_numpy()),
        "interval_of_day": rep(iod), "hour": rep(iod // 6), "minute": rep((iod % 6) * 10),
        "weekday": rep(cal["weekday"].to_numpy()), "month": rep(cal["month"].to_numpy()),
        "day_of_year": rep(cal["day_of_year"].to_numpy()), "iso_week": rep(cal["iso_week"].to_numpy()),
        "season": rep(cal["season"].to_numpy()), "is_weekend": rep(cal["is_weekend"].to_numpy()),
        "is_holiday": rep(cal["is_holiday"].to_numpy()), "is_business_hours": rep(business), "is_night": rep(iod < 36),
        "source": rep(cal["source"].to_numpy()), "source_date": rep(cal["source_date"].to_numpy()),
        "cell_id": np.tile(cells, steps), "zone": np.tile(cells_tbl["zone"].to_numpy(), steps),
        "policy": flat(np.array(POLICIES)[st("pol")]), "idle_share": flat(r3(action[:, :, 3])),
        "event_type_code": flat(st("code")), "event_type": flat(np.array(CODE_NAMES)[st("code").astype(int)]),
        "event_id": flat(st("eid")), "anomaly": flat((st("code") > 0) & (st("code") != CELL_SLEEP)),
        "cell_asleep": flat(asleep),
        "capacity_factor": flat(r3(cf)), "interference_factor": flat(r3(st("kf"))), "capacity_mbps": flat(r3(cap)),
        "spare_capacity_mbps": flat(r3(cap - tot_served)), "total_demand_mbps": flat(r3(d.sum(axis=2))),
        "total_served_mbps": flat(r3(tot_served)), "prbs_used": flat(r3(np.minimum(PRBS, tot_served / cap * PRBS))),
        "load": flat(r3(load)), "neighbour_load": flat(r3(st("nbr_load"))),
        "neighbour_mean_demand_mbps": flat(r3(nbr_dem)), "neighbour_all_sla_ok": flat(r3(nbr_ok)),
        "sinr_db": flat(r3(sinr)), "cqi": flat(cqi), "modulation": flat(MODULATION[cqi]),
        "spectral_efficiency_bps_hz": flat(CQI_EFFICIENCY[cqi]),
        "power_w": flat(r3(pw)), "energy_wh": flat(r3(pw / 6.0)),
        "energy_efficiency_mbit_per_j": flat(r3(np.where(pw > 0, tot_served / np.maximum(pw, 1e-6), 0.0))),
        "rehomed_in_mbps": flat(r3(st("r_in"))), "rehomed_out_mbps": flat(r3(st("r_out"))),
        "handover_users_est": flat(r3(st("r_in") / USER_RATE_MBPS[0])),
        "reconfiguration_churn": flat(r3(inf("churn"))), "anomaly_score": flat(r3(st("z"))),
        "all_slices_sla_ok": flat(all_ok.astype(bool)),
        "reward": flat(r3(st("reward"))), "team_reward": flat(r3(st("team"))),
    }
    tgt = np.array(cfg.target_latency_ms, np.float32)
    per = {"demand_mbps": d, "demand_next_10min_mbps": st("f1"), "demand_next_30min_mbps": st("f3"),
           "demand_next_60min_mbps": st("f6"), "demand_1h_mean_mbps": st("d1h"), "demand_delta_mbps": st("delta"),
           "users_est": users, "share": action[:, :, :3], "prbs": action[:, :, :3] * PRBS, "allocated_mbps": alloc,
           "utilisation": np.minimum(inf("utilisation"), 1e3), "served_mbps": served,
           "throughput_per_user_mbps": np.where(users > 0, served / np.maximum(users, 1e-9), 0.0),
           "packets": served * 1e6 * 600 / (8 * PACKET_BYTES), "queue_kbit": served * p50,
           "latency_p50_ms": p50, "latency_p95_ms": p95, "jitter_ms": jit, "loss": loss,
           "sla_margin_ms": tgt - p95, "sla_ok": ok}
    for field, arr in per.items():
        for s, sl in enumerate(SLICES):
            v = flat(arr[:, :, s])
            if field == "sla_ok":
                cols[f"{sl}_{field}"] = v
            elif field == "loss":
                cols[f"{sl}_{field}"] = v.astype(np.float32)
            elif field == "packets":
                cols[f"{sl}_{field}"] = np.rint(v).astype(np.int64)
            else:
                cols[f"{sl}_{field}"] = r3(v)
    rows = steps * n
    missing = rng.random(rows) < np.where(flat(outage | asleep), 0.5, 0.01)
    noisy = lambda v, s: np.where(missing, np.nan, v * (1 + rng.normal(0, s, rows))).astype(np.float32).round(3)
    cols["obs_total_demand_mbps"] = noisy(cols["total_demand_mbps"], 0.05)
    cols["obs_embb_latency_p95_ms"] = noisy(cols["eMBB_latency_p95_ms"], 0.05)
    cols["obs_load"] = noisy(cols["load"], 0.03)
    obs_sinr = np.where(missing, np.nan, cols["sinr_db"] + rng.normal(0, 1.0, rows)).astype(np.float32).round(2)
    cols["obs_sinr_db"] = obs_sinr
    cols["obs_cqi"] = np.where(missing, -1, cqi_from_sinr(np.nan_to_num(obs_sinr, nan=-20))).astype(np.int8)
    cols["telemetry_missing"] = missing
    table = pa.Table.from_pandas(pd.DataFrame(cols), preserve_index=False)
    key = int(day_index[0]) if per_file == "day" else int(day_index[0]) // 7
    if writer["key"] != key:
        if writer["w"] is not None:
            writer["w"].close()
        path = out_dir / "timeseries" / name / (f"day_{key:03d}.parquet" if per_file == "day" else f"week_{key:02d}.parquet")
        path.parent.mkdir(parents=True, exist_ok=True)
        writer["w"] = pq.ParquetWriter(path, table.schema, compression="zstd")
        writer["key"] = key
    writer["w"].write_table(table.cast(writer["w"].schema))
    stats["rows"] += rows
    stats["sla"] += ok.reshape(-1, 3).sum(axis=0)
    stats["reward"] += float(st("reward").sum())
    stats["energy_kwh"] += float(pw.sum() / 6.0 / 1000)
    stats["served_tb"] += float(tot_served.sum() * 600 / 8 / 1e6)
    stats["asleep"] += int(asleep.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, default=20, help="side of the centred cell block; 100 = the whole city")
    ap.add_argument("--days", type=int, default=364)
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    region = "city" if args.block == GRID else f"core{args.block}"
    # The standard core year keeps the top-level folders; every other build gets its own subfolder.
    sub = "" if (args.block, args.days) == (20, 364) else f"{region}_{args.days}d"
    out_dir, commit = BASE / sub, COMMIT_ROOT / sub
    out_dir.mkdir(parents=True, exist_ok=True); commit.mkdir(parents=True, exist_ok=True)
    print("real Milan grid cache ...", flush=True)
    grid = full_grid(BASE / "milan_full_grid_62d.npy")
    cells = block_cells(args.block) if args.block < GRID else list(range(1, GRID * GRID + 1))
    idx = [c - 1 for c in cells]
    nbrs = grid_neighbours(cells)
    calendar = build_calendar(args.days, args.seed)
    tbl = cell_table(cells, nbrs, args.seed)
    # One Mbps scale for every build, fitted on the dense central 20 x 20 block (the core dataset),
    # so a cell has the same demand in the core and city builds and outer cells are lightly loaded,
    # as suburban macro cells are.
    rng = np.random.default_rng(args.seed + 5)
    core = [c - 1 for c in block_cells(20)]
    sample = np.concatenate([np.asarray(grid[t])[core] for t in np.sort(rng.integers(0, REAL_DAYS * PER_DAY, 3000))])
    scale = (np.array(ModelConfig().demand_p95_fraction) * ModelConfig().capacity_mbps
             / np.maximum(np.percentile(sample, 95, axis=0), 1e-9)).astype(np.float32)
    events = schedule_events(args.days, args.block, args.seed)
    tbl.to_csv(commit / "cells.csv", index=False)
    calendar.to_csv(commit / "calendar.csv", index=False)
    events.assign(cells=events["cells_local"].map(lambda ls: ";".join(str(cells[i]) for i in ls)),
                  params=events["params"].map(json.dumps)).drop(columns=["cells_local"]).to_csv(commit / "events.csv", index=False)
    summary_path = commit / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    summary.update({"region": region, "cells": len(cells), "days": args.days, "intervals": args.days * PER_DAY,
                    "demand_scale": scale.tolist(), "events_per_scenario": events["type"].value_counts().to_dict(),
                    "output": str(out_dir)})
    summary.setdefault("scenarios", {})
    T = args.days * PER_DAY
    per_file = "day" if len(cells) > 2000 else "week"
    for name in args.scenarios.split(","):
        print(f"scenario {name} ({len(cells)} cells x {T} intervals) ...", flush=True)
        prov = DayProvider(name, grid, idx, calendar, scale, None if name == "baseline" else events, nbrs, tbl, args.seed)
        summary["scenarios"][name] = run_scenario(name, prov, cells, nbrs, tbl, calendar, T, args.seed, out_dir, per_file)
        print(f"  {name}: {json.dumps({k: v for k, v in summary['scenarios'][name].items() if k != 'by_event_type'})}", flush=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("done", flush=True)


if __name__ == "__main__":
    main()
