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
      **Confirmed at scale**: scanned 200 real slice files (4,350 real slice
      records) from the full B5G release — `latency_ms`, `jitter_ms`, and
      `packet_loss_rate` are **null in every single one**. This isn't a partial
      gap to wire around; the dataset genuinely never populates these fields, so
      the reward can't be extended with real B5G data alone. Would need either a
      different data source or a live simulator (Simu5G, once wired in) to supply
      these terms.
- [x] Ran the full evaluation suite (`marl.evaluation.evaluate_b5g`) against 50 real
      samples from the full B5G release on the E: drive (11,783
      real agents). **Re-run after fixing an environment bug** (see the MADDPG
      item below — earlier numbers here, 0.3611/0.3674, were computed against
      mis-attributed slices and are superseded). Corrected result
      (`reports/krish_marl_50/`): equal-split 0.3687, heuristic 0.4287, untrained
      actor 0.3660 mean reward; utilization 0.333 / 0.533 / 0.324. The heuristic now
      beats equal-split by 0.06, as expected. Approval rate is still 0% (B5G has no
      jitter/packet-loss telemetry, so the SLA check resolves to "unavailable").
- [x] Sync with Thrishala on the exact format of the TGNN's predicted-demand output
      — confirmed settled, not just drafted: `interface_contracts.md` §2's
      `DemandPrediction` fields (`entity_id`, `slice_type`, `predicted_throughput_bps`,
      `predicted_latency_ms`, `predicted_load`, `confidence`, `horizon_start`/`horizon_end`)
      match `b5g_env.py`'s dataclass exactly, and `orchestrator.py` already
      constructs a real `DemandPrediction` per agent and assigns it to
      `AgentObservation.prediction` in the live closed loop — this is wired in,
      not still pending.
- [x] Implement and train MADDPG against `B5GSlicingEnv` with real gradient updates.
      `MADDPGTrainer` (`src/marl/maddpg.py`): actor + centralized critic, MSE critic
      loss, deterministic policy-gradient actor loss, real `.backward()`/`optimizer.step()`.

      **Environment bug found and fixed (this invalidated every earlier MARL number).**
      B5G's slice `number` restarts for each slice type (298 of 300 sampled files
      reuse numbers), but `b5g_env.py` looked slices up by number in `step()`, so most
      agents were scored against the first slice with that number (usually eMBB): wrong
      slice type and wrong `delta` in the reward. It surfaced when a tuned policy
      "beat" the baselines with only 4% own-slice utilization by allocating 100% to eMBB
      for everyone. Fix: agents are keyed by slice-list position with an explicit
      agent-to-slice map; regression test added
      (`test_agents_with_reused_slice_numbers_are_scored_against_their_own_slice`).
      Agent counts were unchanged (11,783), so the damage was reward mis-attribution,
      not dropped agents. Superseded and removed: the 400-episode (0.340), 2,000-episode
      (0.309) and first tuning results, and the old training curve.

      **Corrected result** (`src/marl/tune_maddpg.py`, `reports/krish_marl_tuned/`).
      Protocol: train on samples 0-999, select on validation block 1000-1049, report once
      on a fresh test block 3000-3049 (15,000+ agents, never used for selection).
      Trained actor **0.4474** vs. heuristic 0.3722 vs. equal-split 0.3122, with own-slice
      utilization 0.784. Training curve (`docs/figures/maddpg_training_curve.png`):
      per-episode reward 0.332 (first 50) -> 0.465 (last 50), above the heuristic's 0.413 on
      the same samples.

      **Caveats to keep attached to that result:** (1) the reward is nearly trivial —
      latency/packet-loss/SLA terms are inactive (null data), the QoS term ignores the
      action, and churn is a constant, so the only actionable signal is "give your own
      slice more share"; the policy beating the heuristic mostly reflects that, not
      sophisticated slicing. (2) All 8 hyperparameter configs land within 0.001 of each
      other (validation 0.4576-0.4583), so the search is uninformative, and `target_tau`
      has no effect (episodes are one-step with `done=True`, so target networks are never
      used). (3) 50 test samples, single seed. Extending the reward (item above) is what
      would make this comparison meaningful.
