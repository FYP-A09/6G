# Dataset catalog

Raw data is gitignored (see `../../.gitignore`) — **except** the five small
(<20MB) slice-labeled datasets marked "in repo" below, which are committed directly
so every teammate has them with a plain `git pull`, no separate download step. This
file is the source of truth for what each dataset is, where it comes from, and how
to re-fetch it. Use `src/data/download_datasets.py` to reproduce everything below.

**Storage note (17 Sep 2026):** the dev machine's C: drive doesn't have room for the
large datasets (NeversNet5G full is 28GB uncompressed, Milan full is ~21GB). Both are
stored in full at `E:\FYP DATA\6G\data\raw\` on that machine instead — outside the
git repo, since they're far too large to commit. Per-person zips of everything that
couldn't go to git are also being shared via Drive (see `docs/team/<person>/tasks.md`
for who owns which dataset).

## In the repo (small, git-committed)

| Folder | Dataset | Size | Fetch command |
|---|---|---|---|
| `robertbotez_6g_slicing/` | 6G synthetic slice-classification CSV (Botez et al., *Electronics* 2025) | 1.2MB | `download_datasets.py robertbotez` |
| `deepslice_secure5g/` | DeepSlice & Secure5G slice-type + QoS dataset (Kaggle) | 76KB | `download_datasets.py kaggle` |
| `network_slicing_puspakmeher/` | Slice-type-labeled QoS dataset (Kaggle, MIT license) | 4.1MB | `download_datasets.py kaggle` |
| `network_slicing_5g_amohankumar/` | Slice-type-labeled QoS dataset, train/test split (Kaggle, CC0) | 2.4MB | `download_datasets.py kaggle` |
| `6g_ran_telemetry_fl/` | 6G RAN federated-learning telemetry — 200 simulated clients, per-slice probabilities (eMBB/URLLC/mMTC/XR/BE), SLA flags, reward column already computed (Kaggle, CC0) | 12MB | `download_datasets.py kaggle` |

## Downloaded, but too large for git (gitignored, kept on `E:\FYP DATA\6G\`, shared via Drive as per-person zips)

Ownership here follows a **full-data vs. sample-data** split: whoever's task needs
the complete dataset works directly from `E:\FYP DATA\6G\` (Keerthivasan's machine —
no need to zip/transfer tens of GB), while whoever only needs enough data to
prototype gets a small zip via Drive instead.

| Folder | Dataset | Full size | Sample given via Drive | Owner (full) | Owner (sample/prototyping) |
|---|---|---|---|---|---|
| `5g_nidd/` | 5G-NIDD real-testbed intrusion-detection flow records (UCD NetSlab, 2022) — single 263MB CSV, over GitHub's 100MB/file limit | 263MB | Whole thing (`Sriranjana_C.zip`, 31MB) | — | Sriranjana C |
| `b5g_slicing/` | B5G Network Slicing Dataset — topology + eMBB/URLLC/mIoT routing/QoS, over/under-provisioned scenarios (Farreras et al., 2024) | 6.6GB | Whole thing (`Krish_S.zip`, 270MB) | — | Krish S |
| `milan_telecom_italia_full/` | Telecom Italia Milan, full 62-day span — Kaggle mirror of the exact Harvard Dataverse files, **no Guestbook form required** (unlike the original Dataverse page) | ~20GB | 1-week sample, 7 daily CSVs (`Thrishala_S_N_partial.zip`, 643MB) | Keerthivasan | Thrishala S N |
| `neversnet5g/` | Full NeversNet5G vehicular 5G NR dataset — 8 part-folders, 242M+ rows of per-UE SINR/CQI/throughput/latency/mobility on a real gNB layout | ~28GB unpacked | 1 part-folder (`part1`+`part1_5`, ~6.4GB raw) via `Thrishala_S_N_neversnet5g_sample.zip` | Keerthivasan | Thrishala S N |

Fetch commands: `download_datasets.py kaggle` (5G-NIDD), `download_datasets.py b5g`
(B5G), `kaggle datasets download -d dkgmgo/telecom-italia-milan` (Milan full),
`download_datasets.py neversnet5g_full` (NeversNet5G full) — or `neversnet5g_sample`
for just a part-folder.

## Recommended use by pipeline stage

No single dataset here provides unlabelled telemetry + real topology + slice-tagged
QoS + closed-loop control all at once, so combine by stage:

- **SSL pretraining corpus** (volume + realism): Milan + `5g_nidd/`.
- **TGNN topology & dynamics** (graph-shaped, time-varying): NeversNet5G (gNB/UE graph with SINR, CQI, throughput, latency over time) and `b5g_slicing/graphs/` (GML topology files).
- **MARL reward shaping / slice-type supervision**: `b5g_slicing/slices/` (over/under-provisioned eMBB/URLLC/mIoT) + `deepslice_secure5g/`, `network_slicing_puspakmeher/`, `network_slicing_5g_amohankumar/` for slice-type labels + `6g_ran_telemetry_fl/` which already ships a `reward` and `sla_ok_next` column per client-window.
- **Missing piece**: no dataset here pairs real topology with time-varying per-slice resource-allocation *actions*. Plan to generate that via Simu5G/ns-3 5G-LENA wrapped as a PettingZoo/RLlib Dec-POMDP (see main literature notes).
