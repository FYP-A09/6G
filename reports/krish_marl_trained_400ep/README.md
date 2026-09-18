# Krish MARL Evaluation

Streaming B5G replay comparison for equal-split, heuristic, and policy actions.
The B5G release exposes the recorded `delta`; missing latency and packet-loss fields are reported as zero.

| Policy | Samples | Mean reward | QoS satisfaction | SLA violation | Utilization | Churn | Inference ms/sample |
|---|---:|---:|---:|---:|---:|---:|---:|
| equal_split | 50 | 0.3893 | 0.0000 | 0.0000 | 0.3333 | 0.2500 | 2.603 |
| heuristic | 50 | 0.3896 | 0.0000 | 0.0000 | 0.5333 | 0.2500 | 2.927 |
| trained_actor | 50 | 0.3400 | 0.0000 | 0.0000 | 0.2470 | 0.2500 | 10.572 |

Artifacts: `summary.csv`, `summary.json`, and `policy_comparison.png`.
Threshold calibration remains provisional until the source paper's exact `delta` definition is verified.
