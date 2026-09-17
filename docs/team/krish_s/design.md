# Design — MARL / Dynamic Slicing (Krish S)

## Problem formulation
A Partially Observable Stochastic Game / Dec-POMDP across *N* slices and *M* base
stations — matches both the RAN-POSG framing in Li et al.'s SF-DTMC paper and the
POMDP framing in the vehicular-network MADDPG paper.

## State
`[current per-slice PRB utilization, TGNN-predicted per-slice demand for the next
window, current QoS metrics (latency/jitter/loss), previous action]`

The TGNN prediction term is the key addition versus every baseline paper reviewed —
it's what makes the controller predictive instead of reactive.

## Action space
Continuous (or discretized) per-slice PRB/bandwidth share, matching the DDPG/MADDPG
continuous-action approach used in the Ali & Arslan NTN paper and the vehicular MARL
paper.

## Reward
```
R = w1·QoS_satisfaction + w2·resource_utilization
  − w3·latency_penalty − w4·packet_loss_penalty
  − w5·SLA_violation_penalty − w6·reconfiguration_penalty
```
The `reconfiguration_penalty` term is borrowed from SF-DTMC's Adaptive Retention
Framework idea — it discourages allocating on every tick just because it can.

## Algorithm candidates
- **MADDPG** — continuous actions, proven in the vehicular slicing paper.
- **SF-DTMC-style truncated Monte Carlo** — better scalability per Li et al. (2026).

Decide between them after prototyping against the B5G environment (see Tasks).

## Environment wrapper
A PettingZoo `ParallelEnv` wrapping either:
1. Offline replay of B5G Slicing dataset scenarios (available now), or
2. A live Simu5G / ns-3 5G-LENA simulation (once Keerthivasan's DT/simulator task lands).

## Training safety
All policy updates are validated inside the Digital Twin before promotion to a
physical-rollout candidate — this is Keerthivasan's DT sync design; this module only
proposes actions, it never applies them directly.
