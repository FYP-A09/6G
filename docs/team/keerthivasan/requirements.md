# Requirements — Digital Twin + Integration (Keerthivasan)

## Functional
- **FR1** Maintain a synchronized virtual replica of physical network state
  (topology, traffic, slice allocations) for safe testing of MARL policies before
  physical rollout — the core NDT requirement from the Native NDT architecture paper.
- **FR2** Support the full NDT lifecycle: Preparation, Creation, Runtime, Feedback
  (per the Native NDT paper's 4-stage model).
- **FR3** Integrate the three modules (SSL embeddings → TGNN predictions → MARL
  actions) into one closed loop, with clear data contracts at each interface.
- **FR4** Support "what-if" analysis (per the GAT virtualization-aware DT paper) —
  test a candidate slice allocation in the twin before committing it physically.

## Non-functional
- **NFR1** Synchronization/update frequency must be fast enough that the twin
  doesn't drift from physical state during a control interval. Most Digital Twin
  papers reviewed do monitoring/simulation *only*, not closed-loop control — this is
  this project's specific differentiator, so it needs to actually work, not just be
  claimed.
- **NFR2** Stay reproducible across every teammate's machine despite large dataset
  sizes — hence the download-script + catalog approach (already built) instead of
  committing multi-GB raw data to git.

## Data / infra requirements
The full multi-dataset catalog (`data/raw/README.md`) and reproducible download
tooling (`src/data/download_datasets.py`) so every teammate can independently
populate their own dataset copy without depending on a shared drive.
