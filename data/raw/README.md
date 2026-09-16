# Dataset catalog

Raw data is gitignored (see `../../.gitignore`) — this file is the source of truth for
what each dataset is, where it comes from, and how to re-fetch it. Use
`src/data/download_datasets.py` to reproduce everything below.

**Disk note (16 Sep 2026):** the dev machine's C: drive was at 98% full (18GB free)
when this catalog was built. Two candidate datasets — the full NeversNet5G archive
(28GB uncompressed) and the full Telecom Italia Milan span (~21GB) — were deliberately
**not** downloaded in full for that reason. Sample/partial versions and download
scripts are provided instead; pull the rest on a machine with more headroom.

## Present on disk (downloaded 16 Sep 2026)

| Folder | Dataset | Size | Fetch command |
|---|---|---|---|
| `robertbotez_6g_slicing/` | 6G synthetic slice-classification CSV (Botez et al., *Electronics* 2025) | 1.2MB | `download_datasets.py robertbotez` |
| `b5g_slicing/` | B5G Network Slicing Dataset — topology + eMBB/URLLC/mIoT routing/QoS, over/under-provisioned scenarios (Farreras et al., 2024) | 6.6GB | `download_datasets.py b5g` |
| `5g_nidd/` | 5G-NIDD real-testbed intrusion-detection flow records (UCD NetSlab, 2022) | 263MB | `download_datasets.py kaggle` |
| `deepslice_secure5g/` | DeepSlice & Secure5G slice-type + QoS dataset (Kaggle) | 76KB | `download_datasets.py kaggle` |
| `network_slicing_puspakmeher/` | Slice-type-labeled QoS dataset (Kaggle, MIT license) | 4.1MB | `download_datasets.py kaggle` |
| `network_slicing_5g_amohankumar/` | Slice-type-labeled QoS dataset, train/test split (Kaggle, CC0) | 2.4MB | `download_datasets.py kaggle` |
| `6g_ran_telemetry_fl/` | 6G RAN federated-learning telemetry — 200 simulated clients, per-slice probabilities (eMBB/URLLC/mMTC/XR/BE), SLA flags, reward column already computed (Kaggle, CC0) | 12MB | `download_datasets.py kaggle` |
| `neversnet5g/` (metadata only) | NeversNet5G vehicular 5G NR dataset — README + metadata docs only, no telemetry rows yet | 88KB | `download_datasets.py neversnet5g_sample` |

## Not yet fetched — script-only

| Target | What it is | Full size | Why deferred |
|---|---|---|---|
| `neversnet5g_full` | Full NeversNet5G: 8 part-folders, 242M+ rows of per-UE SINR/CQI/throughput/latency/mobility on a real gNB layout | ~1.1GB zip / **~28GB unpacked** | Exceeded free disk (18GB). `neversnet5g_sample` in the script pulls just N part-folders instead (~4-7GB per part). |
| `milan_full` | Telecom Italia Milan mobile-traffic grid, Nov 2013–Jan 2014, 100×100 cells | **~21GB** (62 daily files, ~300-380MB each) | Same disk constraint, plus the Dataverse page requires a one-time **Guestbook form** (name/institution/purpose) before any file will download — that's a form submission that needs a human, so it isn't scripted. See `fetch_milan_info()` in `download_datasets.py` for the exact manual steps and the API endpoint pattern to resume with `curl` afterward. |

## Recommended use by pipeline stage

No single dataset here provides unlabelled telemetry + real topology + slice-tagged
QoS + closed-loop control all at once, so combine by stage:

- **SSL pretraining corpus** (volume + realism): Milan (once fetched) + `5g_nidd/`.
- **TGNN topology & dynamics** (graph-shaped, time-varying): NeversNet5G (once fetched — gNB/UE graph with SINR, CQI, throughput, latency over time) and `b5g_slicing/graphs/` (GML topology files).
- **MARL reward shaping / slice-type supervision**: `b5g_slicing/slices/` (over/under-provisioned eMBB/URLLC/mIoT) + `deepslice_secure5g/`, `network_slicing_puspakmeher/`, `network_slicing_5g_amohankumar/` for slice-type labels + `6g_ran_telemetry_fl/` which already ships a `reward` and `sla_ok_next` column per client-window — closest thing here to a ready-made MARL reward signal.
- **Missing piece**: no dataset here pairs real topology with time-varying per-slice
  resource-allocation *actions*. Plan to generate that ourselves via Simu5G/ns-3
  5G-LENA wrapped as a PettingZoo/RLlib Dec-POMDP (see main literature notes).
