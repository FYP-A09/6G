# Requirements — Self-Supervised Learning (Sriranjana C)

## Functional
- **FR1** Learn representations from **unlabelled** telecom telemetry (traffic, flow features, mobility) without requiring manual labels — the core motivation from the SSL survey paper.
- **FR2** Learned embeddings preserve both structural (topology) and content (feature) information per node/flow — Net2Net's specific critique of methods that only capture one or the other.
- **FR3** Output embedding format/dimension is an agreed contract with Thrishala (TGNN owner) so it can be consumed downstream without rework.
- **FR4** The pretext task must be evaluable independently of the downstream task (e.g. via a node/traffic classification proxy), so the SSL stage can be validated in isolation before the TGNN exists.

## Non-functional
- **NFR1** Document the computational overhead of the pretext task — the SSL survey paper explicitly flags "high computational complexity" as an open gap; measure it, don't hand-wave it.
- **NFR2** Generalize across at least two different data sources (e.g. 5G-NIDD real flows + Milan real cellular activity) to avoid overfitting to one testbed's traffic distribution.

## Data requirements
- Large volumes of **real, unlabelled** traffic for pretraining → **5G-NIDD** (`data/raw/5g_nidd/`, real 5G-core flow records) and **Telecom Italia Milan** (`data/raw/milan_telecom_italia/`, real cellular activity grid).
- Slice-type labels *only* for the evaluation/fine-tuning step, never for pretraining → **DeepSlice & Secure5G**, **robertbotez 6G slicing**, and the two **network_slicing_*** Kaggle sets (all already in `data/raw/`).
