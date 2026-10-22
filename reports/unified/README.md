# reports/unified: which files are current

**Current (use these):**

| Topic | Files |
|---|---|
| City forecaster v2 | `forecast_city_v2.json` (accuracy by region against persistence and seasonal baselines), `forecast_value_v2.json` (value to the slicing controllers) |
| MARL vs single agent, held-out week | `marl_v2_scaling_{9,25,100}.json`, `marl_v2_scaling_refs.json` |
| Best MARL, tuned expert, variants | `marl_v2_best.json`, `marl_v2_tuned_expert.json`, `marl_v2_plus.json`, `marl_v2_{ddpg_fc,bc_fc,bc_nofc,refs}.json`, `marl_v2_diagnosis.json` |
| Changed traffic mix | `marl_v2_shift_{experts,ddpg_fc,ddpg_nofc}.json` |
| SSL on 5G-NIDD | `ssl_nidd.json` (masked pretraining), `ssl_nidd_scarf.json` (contrastive) |
| Monitor app data | `showcase_summary.json`, `showcase_city_summary.json` (both use forecaster v2) |
| Dataset and QoS calibration | `dataset_checks.json`, `unified_dataset_sample.csv` |

**Superseded, kept for the record:**

- `results.json`, `single_agent_sweep.json`, `forecast_city.json`: the first unified study (5 training days, 300 episodes, 2 test days) and the
  5-day city forecaster. They under-trained every learner. `forecast_city.json` is still the source of the "10.8x worse than persistence" figure for the
  forecaster trained on 100 cells only.
