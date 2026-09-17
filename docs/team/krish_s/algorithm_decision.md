# MADDPG vs. SF-DTMC-style — algorithm decision

## Comparison

| | MADDPG | SF-DTMC-style (Li et al., 2026) |
|---|---|---|
| Action space | Continuous — natural fit for PRB/bandwidth share | Works with the discretized/statistic-field formulation |
| Proven on | Vehicular slicing paper: 96.5% avg QoS satisfaction | RAN-POSG, large drop in reconfig time + SLA violations vs. LLM-agent and standard RL baselines |
| Scalability | Coordination delay 120-180ms in the vehicular paper — violates URLLC (NFR1) | Explicitly designed to reduce reconfiguration overhead; but paper's own gap says it degrades past tightly-localized clusters |
| Maturity of tooling | Well-supported in PettingZoo/RLlib out of the box | Custom — no off-the-shelf implementation, would need to be built from the paper's description |
| Fit for B5G's offline/replay environment (see `b5g_env.py`) | Straightforward — standard actor-critic loop over recorded episodes | Less natural — SF-DTMC's truncated Monte Carlo assumes online rollouts, harder to adapt to one-shot offline replay |

## Recommendation

**Start with MADDPG.** It has existing library support (RLlib ships a MADDPG-style
trainer), the vehicular-network paper already validates it on a similar
POMDP-shaped slicing problem, and it fits the current B5G offline-replay
environment shape without custom rollout logic. Revisit SF-DTMC-style once (a) a
live Simu5G-backed environment exists (online rollouts become natural) and (b)
MADDPG's scalability limit is actually observed empirically, not just assumed from
the literature.

## Open risk

Both papers report meaningfully different coordination-delay numbers under
different testbeds — neither number should be treated as a guarantee for this
project's specific B5G-based environment. Measure it directly once the environment
in `src/marl/b5g_env.py` produces real reward signals.
