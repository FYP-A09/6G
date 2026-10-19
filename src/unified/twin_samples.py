"""Small committed samples and checks cut from the full Digital Twin dataset (on E:).

Writes to reports/unified_twin/:
  sample_events_day.csv       one recorded day, the 5 x 5 cells around a flash crowd, `events` scenario
  sample_scenarios_cell.csv   one busy cell, one recorded weekday, all six scenarios side by side
  city_62d/sample_night_sleep.csv   a suburban 10 x 10 block, 00:00-06:00, `energy_saving`
  checks.json                 what night-time sleep costs the cells that stay awake, and file sizes

Run: python src/unified/twin_samples.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from unified.env import SLICES  # noqa: E402
from unified.twin_dataset import BASE, COMMIT_ROOT, SCENARIOS  # noqa: E402

CORE, CITY = BASE / "timeseries", BASE / "city_62d" / "timeseries"


def read(folder: Path, filt=None, columns=None) -> pd.DataFrame:
    return ds.dataset(str(folder), format="parquet").to_table(filter=filt, columns=columns).to_pandas()


def folder_gb(p: Path) -> float:
    return round(sum(f.stat().st_size for f in p.rglob("*.parquet")) / 1e9, 2)


def main() -> None:
    out = {}
    # 1. a recorded day around a flash crowd
    ev = pd.read_csv(COMMIT_ROOT / "events.csv")
    fc = ev[(ev.type == "flash_crowd") & (ev.start_interval < 62 * 144)].iloc[0]
    day = int(fc.start_interval // 144)
    members = sorted(int(c) for c in fc.cells.split(";"))
    centre = members[len(members) // 2]
    cells = [centre + dr * 100 + dc for dr in range(-2, 3) for dc in range(-2, 3)]
    f = (ds.field("day_index") == day) & ds.field("cell_id").isin(cells)
    s1 = read(CORE / "events", f).sort_values(["interval", "cell_id"])
    s1.to_csv(COMMIT_ROOT / "sample_events_day.csv", index=False)
    out["sample_events_day"] = {"day_index": day, "event_id": int(fc.event_id), "rows": len(s1), "columns": s1.shape[1]}

    # 2. one busy cell on a recorded weekday, every scenario
    wd = 5  # Wed 6 Nov 2013
    s2 = pd.concat([read(CORE / sc, (ds.field("day_index") == wd) & (ds.field("cell_id") == 5051)) for sc in SCENARIOS])
    s2.sort_values(["scenario", "interval"]).to_csv(COMMIT_ROOT / "sample_scenarios_cell.csv", index=False)
    out["sample_scenarios_cell"] = {"cell_id": 5051, "day_index": wd, "rows": len(s2)}

    # 3. a suburban block at night in the city build, energy saving
    sub = [(r * 100 + c) + 1 for r in range(10, 20) for c in range(10, 20)]
    f3 = (ds.field("day_index") == wd) & (ds.field("hour") < 6) & ds.field("cell_id").isin(sub)
    s3 = read(CITY / "energy_saving", f3).sort_values(["interval", "cell_id"])
    s3.to_csv(COMMIT_ROOT / "city_62d" / "sample_night_sleep.csv", index=False)
    out["sample_night_sleep"] = {"rows": len(s3), "asleep_share": round(float(s3.cell_asleep.mean()), 4)}

    # 4. what sleeping costs: 01:00-05:00, cells that stay awake, against the same rows in baseline (streamed per day file)
    cols = ["interval", "cell_id", "cell_asleep", "rehomed_in_mbps", "load", "energy_wh"] + [f"{s}_sla_ok" for s in SLICES]
    acc = {k: {"rows": 0, "es": np.zeros(3), "base": np.zeros(3), "load_es": 0.0, "load_base": 0.0} for k in ("awake_cells", "awake_cells_absorbing_traffic")}
    energy = {"baseline": 0.0, "energy_saving": 0.0}
    asleep = rows = 0
    for f_es in sorted((CITY / "energy_saving").glob("day_*.parquet")):
        night = (ds.field("hour") >= 1) & (ds.field("hour") < 5)
        es = ds.dataset(str(f_es)).to_table(filter=night, columns=cols).to_pandas()
        base = ds.dataset(str(CITY / "baseline" / f_es.name)).to_table(filter=night, columns=cols).to_pandas()
        assert (es[["interval", "cell_id"]].values == base[["interval", "cell_id"]].values).all()
        awake = ~es.cell_asleep.to_numpy()
        for k, m in (("awake_cells", awake), ("awake_cells_absorbing_traffic", awake & (es.rehomed_in_mbps.to_numpy() > 0))):
            a = acc[k]; a["rows"] += int(m.sum())
            a["es"] += es[[f"{s}_sla_ok" for s in SLICES]].to_numpy()[m].sum(axis=0)
            a["base"] += base[[f"{s}_sla_ok" for s in SLICES]].to_numpy()[m].sum(axis=0)
            a["load_es"] += float(es.load.to_numpy()[m].sum()); a["load_base"] += float(base.load.to_numpy()[m].sum())
        energy["baseline"] += float(base.energy_wh.sum()); energy["energy_saving"] += float(es.energy_wh.sum())
        asleep += int(es.cell_asleep.sum()); rows += len(es)
    cost = {k: {"rows": a["rows"],
                "sla_ok_energy_saving": dict(zip(SLICES, (a["es"] / a["rows"]).round(4).tolist())),
                "sla_ok_baseline": dict(zip(SLICES, (a["base"] / a["rows"]).round(4).tolist())),
                "mean_load_energy_saving": round(a["load_es"] / a["rows"], 4), "mean_load_baseline": round(a["load_base"] / a["rows"], 4)}
            for k, a in acc.items()}
    cost["night_energy_mwh"] = {k: round(v / 1e6, 2) for k, v in energy.items()}
    cost["night_energy_saved"] = round(1 - energy["energy_saving"] / energy["baseline"], 4)
    cost["asleep_share_of_night_rows"] = round(asleep / rows, 4)
    out["energy_saving_cost_city"] = cost

    out["parquet_gb"] = {"core_year": {sc: folder_gb(CORE / sc) for sc in SCENARIOS},
                         "city_62d": {sc: folder_gb(CITY / sc) for sc in ("baseline", "events", "energy_saving")}}
    (COMMIT_ROOT / "checks.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
