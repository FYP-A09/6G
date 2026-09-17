# Tasks — Digital Twin + Integration (Keerthivasan)

**Data split:** you hold the full NeversNet5G (28GB) and full Milan (20GB) on
`E:\FYP DATA\6G\data\raw\` — too large to hand to teammates as zips, so you own
scaling their sample-validated designs up to the full data instead of transferring
it. Thrishala prototypes against a NeversNet5G part-folder sample and a Milan
1-week sample; once her graph schema and TGNN baseline are validated on those, apply
the same logic to the full datasets here and report back whether it holds up at
scale (more nodes, more days, real long-range temporal patterns).

- [ ] Once Thrishala's node/edge schema is validated on the NeversNet5G sample,
      apply it to the full 8-part dataset and check it still holds (schema should
      not have been part1-specific).
- [ ] Once Thrishala's GraphSAGE + temporal-conv baseline works on the samples,
      re-run it against full Milan (62 days) and full NeversNet5G to see how
      generalization (FR3 in her requirements) actually holds at scale.
- [ ] Finalize and circulate the module interface contracts (SSL embedding format,
      TGNN prediction format, MARL action format) so Sriranjana/Thrishala/Krish can
      build against a fixed spec instead of guessing at each other's output shape.
- [ ] Stand up a first Simu5G or ns-3 5G-LENA scenario skeleton (even a toy one) to
      validate that the "Network Element Layer" can actually be synchronized with the
      NDT layer.
- [ ] Keep `data/raw/README.md` and `src/data/download_datasets.py` current as
      teammates pull their datasets.
- [ ] Draft the end-to-end architecture diagram for the review-1 slide, showing all
      three modules inside the Digital Twin loop.
- [ ] Prepare the review-1 slide: your 3 papers + identified gap (monitoring-only
      Digital Twins, no closed-loop optimization) + how this project's DT differs.
