# Tasks — MARL / Dynamic Slicing (Krish S)

- [ ] Pull `data/raw/b5g_slicing/` (already in repo) and inspect the `2.2.1-10kprocessed/slices/` folder structure and QoS fields.
- [ ] Pull `data/raw/6g_ran_telemetry_fl/` (already in repo) and inspect the existing `reward` / `sla_ok_next` columns as a reference reward shape.
- [ ] Draft the Dec-POMDP state/action/reward spec (from `design.md`) and finalize the `w1`–`w6` reward weights.
- [ ] Prototype a PettingZoo environment around the B5G dataset in replay mode (no live simulator needed yet).
- [ ] Decide MADDPG vs. SF-DTMC-style algorithm and document the trade-off.
- [ ] Sync with Thrishala on the exact format of the TGNN's predicted-demand output before wiring it into agent state.
- [ ] Prepare the review-1 slide: your 4 papers + identified gap (reactive control, scalability past local clusters) + how this MARL module addresses it.
