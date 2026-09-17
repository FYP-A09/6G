"""
Full-scale Milan traffic aggregation (Keerthivasan's task — validating that
Thrishala's/Sriranjana's SSL/TGNN feature pipeline actually holds up against the
full 62-day dataset, not just the 1-week sample they prototype against).

Milan's raw schema (data/raw/milan_telecom_italia_full/data/sms-call-internet-mi-*.txt,
tab-separated, no header, confirmed by inspection): square_id, time_interval_ms,
country_code, sms_in, sms_out, call_in, call_out, internet_traffic — 8 columns,
~4.8M rows/day, ~300M rows across all 62 days.

Per docs/architecture/graph_schema_draft.md, Milan isn't graph-structured in the
release itself — it's a feature source (per-cell traffic time series), so this
module's job is producing one aggregated (square_id, time_interval) traffic vector
per row, summed across country codes (roaming breakdown isn't needed for the
traffic-demand signal SSL/TGNN consume), at the full 62-day scale.
"""

from __future__ import annotations

import glob
import os
import time

import pandas as pd

COLUMNS = [
    "square_id", "time_interval_ms", "country_code",
    "sms_in", "sms_out", "call_in", "call_out", "internet_traffic",
]
TRAFFIC_COLUMNS = ["sms_in", "sms_out", "call_in", "call_out", "internet_traffic"]

DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "raw", "milan_telecom_italia_full", "data"
)


def aggregate_one_day(path: str) -> pd.DataFrame:
    """Sums traffic across country codes, keeping (square_id, time_interval_ms)."""
    df = pd.read_csv(
        path, sep="\t", header=None, names=COLUMNS,
        usecols=["square_id", "time_interval_ms"] + TRAFFIC_COLUMNS,
        dtype={"square_id": "int32", "time_interval_ms": "int64"},
    )
    return df.groupby(["square_id", "time_interval_ms"], as_index=False)[TRAFFIC_COLUMNS].sum()


def aggregate_full_dataset(data_dir: str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """
    Runs aggregate_one_day across every daily file found in data_dir and
    concatenates the result — the full-scale run, as opposed to the 1-week
    sample Thrishala/Sriranjana prototype against.
    """
    files = sorted(glob.glob(os.path.join(data_dir, "sms-call-internet-mi-*.txt")))
    if not files:
        raise FileNotFoundError(
            f"No Milan daily files found under {data_dir} — "
            f"run: kaggle datasets download -d dkgmgo/telecom-italia-milan"
        )

    frames = []
    start = time.time()
    for i, path in enumerate(files):
        day_start = time.time()
        frames.append(aggregate_one_day(path))
        print(f"  [{i + 1}/{len(files)}] {os.path.basename(path)} "
              f"({time.time() - day_start:.1f}s)")

    full = pd.concat(frames, ignore_index=True)
    print(f"\nAggregated {len(files)} days in {time.time() - start:.1f}s total")
    return full


if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_DIR
    result = aggregate_full_dataset(data_dir)
    print(f"\nFull-scale result: {len(result):,} (square_id, time_interval) rows, "
          f"{result['square_id'].nunique():,} unique grid cells, "
          f"{result['time_interval_ms'].nunique():,} unique 10-min intervals")
    print(f"Memory footprint: {result.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print(result.head())
