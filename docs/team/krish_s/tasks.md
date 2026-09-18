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

- [~] Calibrate `SLA_DELTA_THRESHOLD` in `b5g_env.py` against the actual Farreras
      et al. (2024) paper's definition of `delta` — empirical groundwork done:
      scanned all 7,984 real slice files under the full B5G release and computed
      the real per-slice-type delta distribution (see
      `reports/krish_marl_50/delta_calibration.md`): eMBB median 0.493 (Q05
      0.049, Q95 0.946), URLLC median 0.500 (Q05 0.050, Q95 0.949), mMTC median
      0.498 (Q05 0.050, Q95 0.950) — deltas are ~uniform on [0,1] for every slice
      type, not concentrated near a natural cutoff. Still open: the paper's own
      definition of what delta value constitutes a violation, which this repo
      doesn't have access to — the threshold itself can't be honestly finalized
      without it.
- [ ] Extend the reward to the remaining 3 terms (latency, packet-loss,
      reconfiguration-churn penalties) once their concrete data sources are
      decided — `delta` alone may not carry enough signal for all of them.
- [x] Ran the full evaluation suite (`marl.evaluation.evaluate_b5g`) against 50 real
      samples from the full `E:\FYP DATA\6G\data\raw\b5g_slicing\` release (11,783
      real agents total, not just the 389 from the 1-sample in-repo fallback).
      Real result: the heuristic policy already beats equal-split on both mean
      reward (0.3674 vs 0.3611) and utilization (0.533 vs 0.333); the untrained
      MADDPG actor's inference cost (6.07ms/sample) is ~3.4x the heuristic's,
      quantifying the real overhead a trained network needs to justify. Approval
      rate is 0% across all 50 samples too \u2014 confirms the missing-telemetry
      finding isn't a fluke of one sample. See `reports/krish_marl_50/`.
- [ ] Sync with Thrishala on the exact format of the TGNN's predicted-demand output
      (already drafted in `interface_contracts.md` §2) before wiring
      `predicted_demand_bps` into `AgentObservation`.
- [ ] Actually implement MADDPG (or an RLlib equivalent) training against
      `B5GSlicingEnv` — the environment exists and runs, the learning algorithm
      doesn't yet.
