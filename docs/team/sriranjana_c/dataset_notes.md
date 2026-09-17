# 5G-NIDD — feature profile (for the SSL pretext task)

`data/raw/5g_nidd/Combined.csv` — 1,215,890 flow records, 50 columns.

## Column groups

- **Flow identity/timing**: `Seq`, `Dur`, `RunTime`, `Offset` — use `Dur`/`RunTime`
  as the "timestep" axis for a masking window; `Seq` as a stable ordering key.
- **Protocol/network**: `Proto`, `sTos`/`dTos`, `sDSb`/`dDSb`, `sTtl`/`dTtl`,
  `sHops`/`dHops`, `Cause`, `State`, `SrcWin`/`DstWin`, `sVid`/`dVid`,
  `SrcTCPBase`/`DstTCPBase` — mostly categorical/protocol-state fields, good
  candidates for embedding lookup rather than direct numeric input.
- **Volume**: `TotPkts`, `SrcPkts`, `DstPkts`, `TotBytes`, `SrcBytes`, `DstBytes`,
  `sMeanPktSz`, `dMeanPktSz` — the core "traffic signal" columns; primary
  candidates to mask-and-reconstruct.
- **Rate/load**: `Mean`, `Sum`, `Min`, `Max`, `Load`, `SrcLoad`, `DstLoad`, `Rate`,
  `SrcRate`, `DstRate` — continuous, already normalized-ish; also good masking
  targets.
- **Loss/latency**: `Loss`, `SrcLoss`, `DstLoss`, `pLoss`, `SrcGap`, `DstGap`,
  `TcpRtt`, `SynAck`, `AckDat` — directly relevant to the eventual QoS/reward
  story, keep these as evaluation-time features even if not masked during
  pretraining.
- **Labels** (fine-tuning/eval only, never pretraining input): `Label`
  (`Benign`/`Malicious`, sampled distribution ~87%/13% malicious in the first
  200k rows — expect class imbalance, stratify any eval split), `Attack Type`,
  `Attack Tool`.

## No inherent graph structure

Unlike NeversNet5G/B5G, 5G-NIDD is flow-level with no gNB/UE topology attached —
use the **flow-level feature encoder** variant from `design.md`, not the
ego-network GNN encoder. If a graph is wanted later, one option is to build a
per-(source, destination) flow graph, but that's out of scope for the first
pretext-task prototype.

## Suggested masking scheme

Mask the **Volume** and **Rate/load** groups (13 columns) per row, keep the
**Protocol/network** columns as always-visible context, and reconstruct the masked
values — this mirrors SSGNN's approach of masking the traffic signal while keeping
structural/context features intact.
