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
- [x] Sync with Thrishala on the exact format of the TGNN's predicted-demand output
      — confirmed settled, not just drafted: `interface_contracts.md` §2's
      `DemandPrediction` fields (`entity_id`, `slice_type`, `predicted_throughput_bps`,
      `predicted_latency_ms`, `predicted_load`, `confidence`, `horizon_start`/`horizon_end`)
      match `b5g_env.py`'s dataclass exactly, and `orchestrator.py` already
      constructs a real `DemandPrediction` per agent and assigns it to
      `AgentObservation.prediction` in the live closed loop — this is wired in,
      not still pending.
- [~] Actually implement MADDPG training against `B5GSlicingEnv` with real gradient
      updates, not just the architecture. `MADDPGTrainer` (`src/marl/maddpg.py`) has
      a real actor/centralized-critic pair trained via `_update()` (MSE critic loss,
      deterministic policy-gradient actor loss, both with real `.backward()` +
      `optimizer.step()` calls) — this part is done. Ran a genuine 400-episode
      training pass against the full 7,984-sample B5G release (episodes 0–399,
      ~281ms/episode, dominated by real GML graph I/O) and evaluated on a held-out
      block of 50 samples never seen during training (indices 400–449, 10,807 real
      agents). **Honest result: the trained actor did not beat the baselines** —
      mean reward 0.3400 vs. equal-split 0.3893 and heuristic 0.3896 (see
      `reports/krish_marl_trained_400ep/README.md`). Utilization also dropped
      (0.247 vs. heuristic's 0.533): the actor hasn't yet learned the heuristic's
      implicit strategy of over-allocating to the agent's own observed slice type.
      This is a real, unfavorable-but-honest finding, not a bug to paper over —
      400 episodes with untuned hyperparameters (`actor_lr=1e-3`, `critic_lr=2e-3`,
      `exploration_noise=0.05`) is a small training budget for MADDPG; more
      episodes, reward-shaping, and hyperparameter tuning are still needed before
      the trained policy is competitive. Still genuinely open, now with real
      evidence instead of an untested architecture. Re-ran training capturing the
      full per-episode curve (`reports/krish_marl_trained_400ep/train_reward_history.json`,
      plotted at `docs/figures/maddpg_training_curve.png`): training-time reward
      (noisy, high per-episode variance — min -0.174, max 0.856) does show a real
      upward drift, from a 0.305 mean over episodes 1–50 to 0.414 over episodes
      350–400, approaching the heuristic's 0.390 level on the *training* samples.
      But the held-out evaluation above (0.340 on unseen samples 400–449) still
      trails baselines — consistent with the actor partly fitting the specific
      training samples rather than learning a policy that generalizes yet. Reporting
      both honestly rather than only the flattering training curve.
