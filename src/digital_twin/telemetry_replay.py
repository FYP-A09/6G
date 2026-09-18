"""Bounded-memory NeversNet5G telemetry replay for the Digital Twin NEL.

The real simulator configuration is kept in ``simu5g_scenario``.  Until its
external OMNeT++/Simu5G/SUMO toolchain is installed, this module replays the
same event records that the scenario emitted.  It deliberately streams CSV
chunks so the 27-GB source does not need to be materialized in RAM.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd


METRIC_COLUMNS = (
    "sinr_dl_db", "sinr_ul_db", "cqi_dl", "cqi_ul", "throughput_dl_bps",
    "throughput_ul_bps", "latency_ul_ms", "speed", "latitude", "longitude",
)


@dataclass(frozen=True)
class TelemetryEvent:
    """One raw Network Element Layer observation with a stable graph node id."""

    node_id: str
    timestamp_s: float
    metrics: dict[str, float]


class NeversNetTelemetryReplay:
    """Replay event-driven UE telemetry in chronological order per source file."""

    def __init__(self, part_dirs: list[str | Path], chunksize: int = 100_000) -> None:
        if chunksize <= 0:
            raise ValueError("chunksize must be positive")
        self.part_dirs = [Path(directory) for directory in part_dirs]
        self.chunksize = chunksize
        self.files = self._discover_files()
        if not self.files:
            raise FileNotFoundError("No *_ue_*_metrics.csv files found in replay directories")

    def _discover_files(self) -> list[tuple[str, Path]]:
        files: list[tuple[str, Path]] = []
        for directory in self.part_dirs:
            if not directory.is_dir():
                raise FileNotFoundError(f"Replay directory does not exist: {directory}")
            for path in sorted(directory.glob("*_ue_*_metrics.csv")):
                stem = path.name.removesuffix("_metrics.csv")
                _, ue_id = stem.rsplit("_ue_", 1)
                files.append((f"ue_{directory.name}_{ue_id}", path))
        return files

    def iter_events(self, max_events: int | None = None) -> Iterator[TelemetryEvent]:
        """Yield finite metric observations, stopping after ``max_events`` if set."""
        emitted = 0
        wanted = ["time_s", *METRIC_COLUMNS]
        for node_id, path in self.files:
            for frame in pd.read_csv(path, usecols=lambda column: column in wanted, chunksize=self.chunksize):
                if "time_s" not in frame:
                    continue
                frame = frame.sort_values("time_s")
                for row in frame.itertuples(index=False):
                    values = row._asdict()
                    timestamp = pd.to_numeric(values.pop("time_s"), errors="coerce")
                    if pd.isna(timestamp):
                        continue
                    metrics = {
                        name: float(value)
                        for name, value in values.items()
                        if pd.notna(value)
                    }
                    if not metrics:
                        continue
                    yield TelemetryEvent(node_id, float(timestamp), metrics)
                    emitted += 1
                    if max_events is not None and emitted >= max_events:
                        return

    def latest_state(self, max_events: int | None = None) -> dict[str, TelemetryEvent]:
        """Return the newest observation per UE from a bounded replay pass."""
        state: dict[str, TelemetryEvent] = {}
        for event in self.iter_events(max_events=max_events):
            previous = state.get(event.node_id)
            if previous is None or event.timestamp_s >= previous.timestamp_s:
                state[event.node_id] = event
        return state


__all__ = ["METRIC_COLUMNS", "TelemetryEvent", "NeversNetTelemetryReplay"]
