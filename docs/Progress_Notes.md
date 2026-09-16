# Progress Notes

## Minutes of Meeting — 11 July 2026 (Kickoff)

**Project:** Digital Twin Assisted Dynamic Network Slicing using SSL, TGNN and MARL for AI-Native 6G Networks

### Agenda
- Review project problem statement and proposed solution
- Discuss research gap and novelty
- Define literature survey expectations
- Plan publication roadmap
- Discuss dataset and simulation requirements

### Discussion Points

**1. Project Overview** — Digital Twin Assisted Dynamic Network Slicing for AI-Native 6G Networks, targeting AR/VR, autonomous vehicles, smart healthcare, and massive IoT. Current slicing approaches are largely static or semi-dynamic with limited AI-based optimization.

**2. Proposed Methodology** — Digital Twin as a virtual replica for simulation/testing/optimization; SSL to learn traffic representations from unlabeled telecom data; TGNN to predict future network states; MARL to manage resource allocation across eMBB, URLLC, and mMTC; dynamic slicing driven by the predicted conditions.

**3. Research Gap and Novelty** — Existing literature already covers Digital Twin+RL, MARL for slicing, and GNN/TGNN traffic prediction individually. Novelty is positioned around (a) integrating SSL before TGNN to learn from unlabeled telecom data, (b) improving forecasting using SSL-derived representations, (c) moving from reactive to predictive slicing. Faculty advised novelty claims must be backed by literature evidence, not assumption.

**4. Literature Survey Plan** — Focus primarily on IEEE Transactions papers (Springer/Elsevier acceptable); cover the last six years; ~5–6 papers per member; gaps must be justified through published work.

**5. Documentation Process** — Maintain a common project notebook/document for the team; record contributions; present updates at every review.

**6. Publication Roadmap** — Target: 1 survey paper, 2 journal papers, 1 conference paper; patent filing possible if novelty holds. Suggested venues: IEEE GLOBECOM, IEEE VTC.

**7. Dataset and Simulation Requirements** — Use realistic, multi-class datasets; prefer real-world data; if generated, follow established protocols; avoid oversimplified network models.

### Action Items (11 Jul 2026)
- Each member collects 5–6 papers related to network slicing.
- Conduct literature survey and identify research gaps.
- Maintain a shared progress document.
- Prepare literature-backed novelty justification.
- Explore suitable datasets and simulation environments.
- Develop a detailed research roadmap.
- Begin planning the survey paper.

### Faculty Remarks
Strong literature review is essential before implementation; novelty must be evidence-backed; maintain research consistency; favor realistic datasets and high-fidelity simulation; sequence publications so one leads to the next.

---

## Progress Update — 16 September 2026

Status against the 11 Jul action items, plus dataset acquisition carried out this session.

### 1. Literature survey
- **16 papers logged** in [`Literature_Review_Notes.md`](Literature_Review_Notes.md) across two contributors, spanning IEEE Transactions on Mobile Computing, IEEE Wireless Communications Letters, IEEE Transactions on Intelligent Transportation Systems, TechRxiv, arXiv, and journal/preprint venues from 2022–2026.
- Coverage across all four pillars: Digital Twin (5 papers), SSL (3 papers), GNN/TGNN for slicing (3 papers), MARL/DRL for slicing (5 papers).
- Confirmed novelty position: no surveyed paper combines SSL + TGNN + Digital Twin + MARL for 6G slicing. The recurring gap across the Digital-Twin/DRL papers specifically is **reactive, not predictive** decision-making with no representation-learning stage — this directly motivates inserting SSL→TGNN ahead of the MARL controller.
- Remaining gap in coverage: 6G-specific (not 5G) MARL slicing papers with real-hardware validation are still thin — continue searching for these.

### 2. Dataset shortlist — now downloaded, not just identified
Repo created at `github.com/FYP-A09/6G`; the following are now on disk under `data/raw/` (catalog + fetch commands in [`../data/raw/README.md`](../data/raw/README.md)):

| Dataset | Status | Role |
|---|---|---|
| B5G Network Slicing Dataset | ✅ downloaded & extracted (6.6GB, 24k files: graphs/routings/slices) | TGNN topology + MARL reward/slice-type benchmark |
| 5G-NIDD | ✅ downloaded (263MB) | SSL pretraining (real 5G-core flow features) |
| DeepSlice & Secure5G | ✅ downloaded | Slice-type classification benchmark |
| Network Slicing (puspakmeher, amohankumar) | ✅ downloaded | Slice-type classification benchmark |
| 6G RAN Telemetry for Federated Learning | ✅ downloaded (12MB) — already has per-slice probabilities, SLA flags, and a computed `reward` column | Closest existing proxy for a MARL reward signal |
| robertbotez 6G slicing dataset (2025) | ✅ downloaded (1.2MB) | Secondary 6G-labeled benchmark |
| NeversNet5G (vehicular 5G NR) | ⚠️ metadata only — full dataset is 28GB uncompressed | TGNN topology & dynamics (gNB/UE graph, time-varying SINR/CQI/throughput/latency) |
| Telecom Italia Milan | ⚠️ not fetched — ~21GB, and Harvard Dataverse requires a manual Guestbook form before any file downloads | SSL pretraining corpus (real, large-scale, unlabelled) |

**Blocker hit and resolved:** the dev machine's C: drive was at 98% full (18GB free, unrelated to this project) when the NeversNet5G extraction was attempted — it failed mid-extraction needing 28GB. Decision (confirmed with the team lead): keep the working set lightweight for now: download scripts for the two large/gated datasets are committed (`src/data/download_datasets.py`), to be run on a machine with more free disk rather than filling this one.

### 3. Open gaps before implementation
- No dataset pairs real topology with time-varying, per-slice resource-allocation *actions and rewards* together — this piece will need to be generated via simulation (Simu5G / ns-3 5G-LENA) rather than sourced.
- SSL pretext task not yet fixed — candidates from the literature review are masked traffic reconstruction (SSGNN-style), contrastive learning, and position-prediction (Net2Net-style). Needs a decision before dataset preprocessing can be finalized.
- TGNN library (PyTorch Geometric Temporal) and MARL environment (PettingZoo + Ray RLlib) are the working choices from the literature review but not yet integrated into code.

### Next Steps
- [ ] Fetch Milan (manual Guestbook step) and full/partial NeversNet5G on a machine with ≥30GB free.
- [ ] Decide and document the SSL pretext task.
- [ ] Stand up a first Simu5G or ns-3 5G-LENA scenario to test whether it can jointly emit topology + slice-tagged QoS.
- [ ] Continue literature survey toward hardware-validated 6G MARL slicing papers.
- [ ] Begin outlining the survey paper — Related Work can draw directly from the 16 logged reviews.

### Follow-up needed from advisor
Confirm whether "no public dataset covers the full pipeline, so the RAN+MARL closed loop is simulated" is an acceptable methodology framing for the survey paper, or whether reviewers will expect at least one fully-real end-to-end validation.
