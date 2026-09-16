"""
Dataset fetcher for the Digital-Twin-Assisted Dynamic Network Slicing project.

Each dataset lives under data/raw/<name>/. Run with one or more dataset names,
or `all` for everything that fits comfortably on disk (excludes the two large
ones — neversnet5g_full and milan_full — which you must opt into explicitly
once you have enough free space; see data/raw/README.md for sizes).

Usage:
    python src/data/download_datasets.py robertbotez b5g kaggle
    python src/data/download_datasets.py neversnet5g_sample
    python src/data/download_datasets.py all

Requires: curl on PATH, and for the `kaggle` target, a configured
~/.kaggle/kaggle.json (see https://github.com/Kaggle/kaggle-api#api-credentials).
"""

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

KAGGLE_DATASETS = {
    "5g_nidd": "humera11/5g-nidd-dataset",
    "deepslice_secure5g": "anuragthantharate/deepslice",
    "network_slicing_puspakmeher": "puspakmeher/networkslicing",
    "network_slicing_5g_amohankumar": "amohankumar/network-slicing-in-5g",
    "6g_ran_telemetry_fl": "nizamuddinmaitlo/wireless-6g-network-dataset-of-resource-allocation",
}


def _curl(url: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["curl", "-fL", "--retry", "3", "-o", str(out), url], check=True)


def fetch_robertbotez() -> None:
    """~1.2MB synthetic 6G slice-classification CSV (Botez et al., Electronics 2025)."""
    out_dir = RAW / "robertbotez_6g_slicing"
    out_dir.mkdir(parents=True, exist_ok=True)
    _curl(
        "https://raw.githubusercontent.com/robertbotez/6g-network-slicing-dataset/"
        "main/network_slicing_dataset%20-%20v3.csv",
        out_dir / "network_slicing_dataset_v3.csv",
    )


def fetch_b5g() -> None:
    """~270MB (6.6GB unpacked) B5G network-slicing simulation dataset (Farreras et al., 2024)."""
    out_dir = RAW / "b5g_slicing"
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("datanetAPI.py", "features.pdf"):
        _curl(f"https://zenodo.org/api/records/10610616/files/{fname}/content", out_dir / fname)
    zip_path = out_dir / "slicing-simulations.zip"
    _curl("https://zenodo.org/api/records/10610616/files/slicing-simulations.zip/content", zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)
    zip_path.unlink()  # already extracted, drop the 270MB archive


def fetch_kaggle() -> None:
    """The five Kaggle-hosted slicing/telemetry/intrusion datasets used in this project."""
    for dir_name, ref in KAGGLE_DATASETS.items():
        out_dir = RAW / dir_name
        out_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", ref, "-p", str(out_dir), "--unzip"],
            check=True,
        )


def fetch_neversnet5g_sample(parts: int = 1) -> None:
    """Metadata + README always; optionally unpack the first `parts` part-folders (~4-7GB each)."""
    out_dir = RAW / "neversnet5g"
    out_dir.mkdir(parents=True, exist_ok=True)
    _curl(
        "https://huggingface.co/datasets/Askedrin/NeversNet5G_Vehicular_5G_NR_Dataset/"
        "resolve/main/metadata_docs.zip",
        out_dir / "metadata_docs.zip",
    )
    _curl(
        "https://huggingface.co/datasets/Askedrin/NeversNet5G_Vehicular_5G_NR_Dataset/"
        "raw/main/README.md",
        out_dir / "README.md",
    )
    if parts <= 0:
        return
    zip_path = out_dir / "data.zip"
    _curl(
        "https://huggingface.co/datasets/Askedrin/NeversNet5G_Vehicular_5G_NR_Dataset/"
        "resolve/main/data.zip",
        zip_path,
    )
    with zipfile.ZipFile(zip_path) as z:
        wanted_prefixes = tuple(f"data/part{i}" for i in range(1, parts + 1))
        members = [n for n in z.namelist() if n.startswith(wanted_prefixes) or n == "data/"]
        z.extractall(out_dir, members=members)
    zip_path.unlink()  # full archive is 28GB uncompressed across 8 parts — don't keep it around


def fetch_neversnet5g_full() -> None:
    """Full ~1.1GB zip / ~28GB uncompressed across 8 part-folders. Requires ~30GB free."""
    free_gb = shutil.disk_usage(RAW).free / 1e9
    if free_gb < 30:
        raise SystemExit(
            f"Only {free_gb:.1f}GB free; NeversNet5G needs ~30GB to download+extract. Aborting."
        )
    out_dir = RAW / "neversnet5g"
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "data.zip"
    _curl(
        "https://huggingface.co/datasets/Askedrin/NeversNet5G_Vehicular_5G_NR_Dataset/"
        "resolve/main/data.zip",
        zip_path,
    )
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)
    zip_path.unlink()


def fetch_milan_info() -> None:
    """
    Telecom Italia Milan grid dataset (Harvard Dataverse, DOI 10.7910/DVN/EGZHFV).
    Gated behind a mandatory Guestbook form (name/institution/purpose) — cannot be
    scripted anonymously. Manual steps:
      1. Visit https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV
      2. Click a file's Download button, fill the one-time Guestbook, accept.
      3. Re-run curl against https://dataverse.harvard.edu/api/access/datafile/<id>
         (the id shown in the file's "Download URL") — it will work after step 2
         because the guestbook acceptance is tied to your session/account.
    62 daily files, ~300-380MB each, ~21GB total for the full Nov 2013-Jan 2014 span.
    Grab a handful of days (a few hundred MB) rather than all 62 unless disk allows it.
    """
    print(fetch_milan_info.__doc__)


TARGETS = {
    "robertbotez": fetch_robertbotez,
    "b5g": fetch_b5g,
    "kaggle": fetch_kaggle,
    "neversnet5g_sample": fetch_neversnet5g_sample,
    "neversnet5g_full": fetch_neversnet5g_full,
    "milan_info": fetch_milan_info,
}

LIGHTWEIGHT = ["robertbotez", "b5g", "kaggle"]


def main(argv: list[str]) -> None:
    if not argv:
        print(__doc__)
        return
    names = LIGHTWEIGHT if argv == ["all"] else argv
    for name in names:
        if name not in TARGETS:
            print(f"Unknown target '{name}'. Options: {', '.join(TARGETS)}")
            continue
        print(f"=== {name} ===")
        TARGETS[name]()


if __name__ == "__main__":
    main(sys.argv[1:])
