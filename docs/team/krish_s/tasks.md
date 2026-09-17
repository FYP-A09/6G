# Tasks — MARL / Dynamic Slicing (Krish S)

## Done (drafted/prototyped ahead of your validation pass — react to these, don't treat them as final)

- [x] B5G dataset inspection — `b5g_dataset_notes.md`. Note: the bundled
      `datanetAPI.py` doesn't match this release's actual flat-file layout
      (confirmed by running it — finds zero samples); `b5g_env.py` reads the real
      files directly instead.
- [x] PettingZoo-shaped offline-replay environment — `src/marl/b5g_env.py`, tested
      end-to-end against the real sample data: loads a 416-node/830-edge B5G graph,
      389 agents, computes real per-agent rewards. Falls back automatically to the
      small in-repo sample when the full 6.6GB dataset isn't unzipped locally.
- [x] Reward formula v1 — 3 of the 6 terms from `design.md` implemented
      (`REWARD_WEIGHTS` in `b5g_env.py`: QoS from `1 - delta`, utilization,
      SLA-violation penalty against a per-slice-type `delta` threshold).
- [x] MADDPG vs. SF-DTMC-style decision — `algorithm_decision.md`, recommends
      MADDPG first.
- [x] Review-1 slide notes — `review1_slide_notes.md`.

## Still yours to do

- [ ] Calibrate `SLA_DELTA_THRESHOLD` in `b5g_env.py` against the actual Farreras
      et al. (2024) paper's definition of `delta` — currently a guessed
      per-slice-type threshold, not confirmed.
- [ ] Extend the reward to the remaining 3 terms (latency, packet-loss,
      reconfiguration-churn penalties) once their concrete data sources are
      decided — `delta` alone may not carry enough signal for all of them.
- [ ] Pull the full `data/raw/b5g_slicing/` (or unzip `Krish_S.zip`) to run against
      all 10k samples instead of the 1-sample in-repo fallback.
- [ ] Sync with Thrishala on the exact format of the TGNN's predicted-demand output
      (already drafted in `interface_contracts.md` §2) before wiring
      `predicted_demand_bps` into `AgentObservation`.
- [ ] Actually implement MADDPG (or an RLlib equivalent) training against
      `B5GSlicingEnv` — the environment exists and runs, the learning algorithm
      doesn't yet.
