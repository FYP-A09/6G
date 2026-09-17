# Review-1 slide notes — MARL / Dynamic Slicing (Krish S)

## Your 4 papers + gap (pull from `docs/Literature_Review_Notes.md`)

1. **Toward Enabling Network Slice Mobility to Support 6G System (2022)** —
   DELAY-NSM/R-NSM/FT-NSM, Kalai-Smorodinsky bargaining game. **Gap**: pure
   optimization/game theory, no learning-based representation or prediction.
2. **Slicing for AI: Online Learning Framework (2025)** — EXP3-based OLS/OLS-SA/
   OLS-RSA. **Gap**: no TGNN prediction, no multi-agent coordination.
3. **Deep Reinforcement Learning for End-to-End Network Slicing (2023)** —
   Safety DRL cuts SLA violations from 15% to ~1%; Distributed/Multi-Agent DRL for
   scalability. **Gap**: purely reactive to observed state.
4. **A Flexible and Scalable MARL Framework for Dynamic RAN Slicing (Li et al.,
   2026)** — SF-DTMC + Adaptive Retention Framework. **Gap**: scalability drops
   past local clusters; still local/reactive observation.

## Your module's answer to the shared gap

Every one of these either skips learned representation entirely (paper 1) or
reacts to the *current* state only (papers 2-4). This project's MARL agents
consume the TGNN's **predicted** future demand as part of their state (see
`docs/architecture/interface_contracts.md` §2) — the reactive→predictive shift is
the whole point, and it plugs directly into the reward-and-action design that
these four papers already validated works for slicing broadly.

## What's built so far

- `src/marl/b5g_env.py` — offline-replay PettingZoo-shaped environment over the
  B5G dataset (real topologies + recorded eMBB/URLLC/mMTC flows + ground-truth
  performance), using the dataset's own `datanetAPI.py` loader rather than
  hand-parsing the JSON.
- `docs/team/krish_s/algorithm_decision.md` — MADDPG chosen as the starting
  algorithm over SF-DTMC-style, with the reasoning and the open risk to test for.
- Reward formula drafted in `design.md`, default weights in `b5g_env.py`
  (`REWARD_WEIGHTS`) — not yet tuned, that's the next concrete step once the
  performance-matrix lookup is wired up.

## Slide checklist

- [ ] 4 papers + gap (above)
- [ ] Reward formula (from `design.md`) with the 6 weighted terms
- [ ] Why MADDPG first (from `algorithm_decision.md`)
- [ ] One diagram slide reused from `docs/architecture/system_diagram.md`, MARL box highlighted
