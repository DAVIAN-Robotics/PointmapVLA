# Relationship to starVLA

PointmapVLA is built on [starVLA](https://github.com/starVLA/starVLA), a
modular codebase for developing vision-language-action models. This repository
keeps starVLA's package name, directory layout and extension points so that
PointmapVLA can be contributed back upstream as a new framework.

* Upstream base: commit `2e5f239bc0b1661d7d556bdba5071f3041544cc6` (2026-07-01).
* The files listed in `UPSTREAM_IDENTICAL.txt` are byte-identical to that
  commit; `tools/check_upstream_parity.sh` verifies this.
* A small number of upstream files carry minimal, documented changes (the
  trainer entry point, the LeRobot loader's depth support, the data-registry
  discovery, the policy server). Everything else PointmapVLA adds lives in new
  files:
  * `starVLA/model/framework/VLM4A/PointmapVLA.py` — the framework
  * `starVLA/model/modules/vlm/QWen3_5_pointmap.py` and
    `starVLA/model/modules/vlm/qwen3_5/` — the pointmap-aware Qwen3.5 interface
  * `starVLA/model/modules/action_model/PointmapVLA_ActionHeader.py` — the
    flow-matching action expert
  * `starVLA/dataloader/gr00t_lerobot/paired_pointmap_dataset.py` and
    `starVLA/dataloader/gr00t_lerobot/transform/pointmap*.py` — pointmap data
  * `examples/simBenchmarks/{Robocasa_kitchen,LIBERO,LIBERO-plus}/` —
    benchmark configs and evaluation clients

Frameworks, benchmarks and modules of upstream starVLA that PointmapVLA does
not use are not included here; please use upstream starVLA for those.

Because the identical files are kept verbatim, they also carry upstream's own
`TODO` comments (for example in
`starVLA/model/modules/action_model/flow_matching_head/cross_attention_dit.py`).
Those are upstream notes, not open items of this release; fixes belong upstream.
