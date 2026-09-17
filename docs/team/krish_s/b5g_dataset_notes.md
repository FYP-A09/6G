# B5G dataset — inspection notes (for the MARL environment)

The dataset ships its own loader, `data/raw/b5g_slicing/datanetAPI.py` (from the
Barcelona Neural Networking Center's RouteNet-family tooling) — **but it expects an
older tar.gz-bundled release layout and doesn't match this dataset's actual
"2.2.1-10kprocessed" flat-file release** (confirmed by running it: `DatanetAPI`
finds zero samples against these files). `src/marl/b5g_env.py` reads the flat
`graphs/graph_N.txt` (GML, via `networkx.read_gml`), `routings/routing_N.txt`
(plain CSV), and `slices/slices_N.json` files directly instead, keyed by the
shared index `N`.

## Folder structure

```
2.2.1-10kprocessed/
  graphs/     graph_<id>.txt   — GML topology (nodes, links, per-link capacity)
  routings/   routing for that graph — path each src/dst pair uses
  slices/     slices_<id>.json — the per-slice flow definitions for that sample
```

## What each sample index `N` gives you (read directly, no API wrapper)

- `graphs/graph_N.txt` — GML topology, loaded with `networkx.read_gml()`. Node
  `type` distinguishes `"inner"` (transport node) from `"antenna"` (radio access
  node) — confirmed by inspection (see `graph_schema_draft.md`).
- `routings/routing_N.txt` — a plain `netSize × netSize` CSV matrix; `-1` on the
  diagonal, otherwise the next-hop port from row-node to column-node.
- `slices/slices_N.json` — a JSON array of slice objects: `type`
  (`eMBB`/`URLLC`/`mMTC`), `delta` (a per-slice deviation value in observed range
  ~0.07–0.48 — treated as a normalized QoS-deviation/SLA-margin signal; not yet
  confirmed against the Farreras et al. 2024 paper's exact definition, calibrate
  the `SLA_DELTA_THRESHOLD` values in `b5g_env.py` once confirmed), `number`
  (slice id), and `flows` (each with `origin_node`, `destination`,
  `origin_node_antenna`, `path`, `bandwidth`, `traffic_string`).

## What this means for the MARL environment

`delta` is a recorded label for a specific already-provisioned configuration, not
something your agent's action changes live — so `src/marl/b5g_env.py` treats B5G
as an **offline replay environment**: one episode = one sample index, the agent
proposes a `slice_allocations` action, and reward combines `(1 - delta)` as a QoS
term with an SLA-violation penalty when `delta` exceeds a per-slice-type
threshold. This is a v1 approximation (3 of the 6 terms in `design.md`'s full
reward formula) — extend with the latency/packet-loss/reconfiguration terms once
their concrete data sources are confirmed.
