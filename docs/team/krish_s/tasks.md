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

## Update 22 Oct 2026: MARL retrained properly on the unified environment

The B5G MADDPG result above is unchanged, but the question "does MARL help?" was re-run on the unified
Milan environment with a real training budget (`src/unified/marl_v2*.py`, full write-up in
`docs/Unified_Data_and_MARL.md` section 2). The first unified study (5 training days, 300 episodes) under-trained
every learner and is superseded.

- Training on 40 days, tested on a held-out week (16-22 Dec), with the TGNN v2 forecast as input.
- **Single agent vs MARL:** single agent 0.44 / 0.34 / 0.28 at 9 / 25 / 100 cells (two learning rates tried);
  MARL 0.93 / 0.83 / 0.86 (1,000 episodes, 2 seeds; 100 cells 1 seed), with a constant 19,332-parameter actor.
- **Against a fair rule baseline:** the expert headroom rule with its two utilisation targets tuned by grid search
  on the training days scores 0.878 at 25 cells. MARL with 4,000 episodes and the checkpoint chosen on validation
  days scores 0.899 / 0.899 / 0.903 (ensemble 0.904): +0.022 to +0.027 over the tuned rule, on 7 of 7 test days.
  With 1,000 to 2,000 episodes and the final checkpoint it only ties the tuned rule.
- Neighbour communication and an expert-residual start added nothing; an expert-regularised actor loss did not help.
- When the traffic mix changes (6G growth, massive IoT) and nothing is retrained, MARL is worse than the tuned rule.
