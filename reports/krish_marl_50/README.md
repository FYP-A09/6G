# Krish MARL Evaluation

Streaming B5G replay comparison for equal-split, heuristic, and policy actions.
The B5G release exposes the recorded `delta`; missing latency and packet-loss fields are reported as zero.

| Policy | Samples | Mean reward | QoS satisfaction | SLA violation | Utilization | Churn | Inference ms/sample |
|---|---:|---:|---:|---:|---:|---:|---:|
| equal_split | 50 | 0.3611 | 0.0000 | 0.0000 | 0.3333 | 0.2500 | 1.753 |
| heuristic | 50 | 0.3674 | 0.0000 | 0.0000 | 0.5333 | 0.2500 | 1.824 |
| untrained_actor_baseline | 50 | 0.3552 | 0.0000 | 0.0000 | 0.3244 | 0.2500 | 6.070 |

Artifacts: `summary.csv`, `summary.json`, and `policy_comparison.png`.
Threshold calibration remains provisional until the source paper's exact `delta` definition is verified.
