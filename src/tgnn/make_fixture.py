"""Generates synthetic event-driven UE CSVs (subset of real UE ids from the
node_mapping file) to validate build_graph.py's windowing/stitching logic
end-to-end, since the real 6.4GB sample isn't available in this sandbox."""
import csv
import random
from pathlib import Path

random.seed(0)

GNB_LAT, GNB_LON = 46.9838, 3.1636  # roughly centered on the real 19-gNB cluster

def write_part1_ue(ue_id, out_path, t_end=366.0):
    rows = []
    t = 0.0
    lat, lon = GNB_LAT + random.uniform(-0.02, 0.02), GNB_LON + random.uniform(-0.02, 0.02)
    while t < t_end:
        t += random.uniform(2, 12)
        if t >= t_end:
            break
        event = random.choice(["mobility", "radio", "app"])
        row = {c: "" for c in HEADER}
        row["time_s"] = round(t, 3)
        row["vehicle_id"] = f"veh_{ue_id}"
        if event == "mobility":
            lat += random.uniform(-0.0005, 0.0005)
            lon += random.uniform(-0.0005, 0.0005)
            row["latitude"] = round(lat, 6)
            row["longitude"] = round(lon, 6)
            row["speed"] = round(random.uniform(0, 15), 2)
        elif event == "radio":
            row["sinr_dl_db"] = round(random.uniform(-5, 30), 2)
            row["cqi_dl"] = random.randint(1, 15)
        else:
            row["throughput_dl_bps"] = random.randint(1_000_000, 50_000_000)
            row["latency_ul_ms"] = round(random.uniform(5, 40), 2)
        rows.append(row)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        w.writerows(rows)
    return lat, lon  # last known position, to seed the transition file


def write_transition_ue(seq_id, out_path, last_lat, last_lon, t_start=366.0, t_end=370.0):
    rows = []
    t = t_start
    lat, lon = last_lat, last_lon
    while t < t_end:
        t += random.uniform(0.5, 1.5)
        if t >= t_end:
            break
        row = {c: "" for c in HEADER}
        row["time_s"] = round(t, 3)
        row["vehicle_id"] = f"veh_transition_{seq_id}"
        lat += random.uniform(-0.0003, 0.0003)
        lon += random.uniform(-0.0003, 0.0003)
        row["latitude"] = round(lat, 6)
        row["longitude"] = round(lon, 6)
        row["sinr_dl_db"] = round(random.uniform(-5, 30), 2)
        rows.append(row)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        w.writerows(rows)


HEADER = [
    "time_s", "vehicle_id", "x", "y", "speed", "edge", "latitude", "longitude",
    "sinr_dl_db", "sinr_ul_db", "cqi_dl", "cqi_ul", "rcvd_sinr_dl_db", "rcvd_sinr_ul_db",
    "rlc_delay_dl_ms", "rlc_delay_ul_ms", "rlc_pdu_delay_dl_ms", "rlc_pdu_delay_ul_ms",
    "rlc_pdu_throughput_dl_bps", "rlc_pdu_throughput_ul_bps",
    "throughput_dl_bps", "throughput_ul_bps", "latency_ul_ms",
]

if __name__ == "__main__":
    # original_ids present in node_mapping_366_370.txt sample (subset)
    original_ids = [3, 4, 5, 6, 7]
    # The fixture is intended to run from a clean checkout, where the large
    # NeversNet5G dataset folders are intentionally absent (and gitignored).
    # Create only the small synthetic directories it owns before writing CSVs.
    Path("data/raw/neversnet5g/data/part1").mkdir(parents=True, exist_ok=True)
    Path("data/raw/neversnet5g/data/part1_5").mkdir(parents=True, exist_ok=True)
    last_positions = {}
    for oid in original_ids:
        path = f"data/raw/neversnet5g/data/part1/part1_ue_{oid}_metrics.csv"
        last_positions[oid] = write_part1_ue(oid, path)
        print(f"wrote {path}")

    # sequential ids from the mapping file for these same original ids: 0,1,2,3,4
    seq_ids = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4}
    for oid, seq in seq_ids.items():
        path = f"data/raw/neversnet5g/data/part1_5/part1_5_ue_{seq}_metrics.csv"
        lat, lon = last_positions[oid]
        write_transition_ue(seq, path, lat, lon)
        print(f"wrote {path}")
