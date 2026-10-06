# 5500M tuning run (2026-09-26/27)

Record of tuning Bonsai 2 27B PTQ1_0 on the Radeon Pro 5500M 8 GB (Vulkan, Windows), branch `5500m-tuning`.

- `SUMMARY.md`: start vs final numbers, why each commit helped, recommended launcher settings, remaining ideas. Start here.
- `RESULTS.md`: every experiment (wins, losses, failures). Read before trying something, so nothing is retried blind.
- `FACTS.md`: Step 0 measurements of the GPU and the start build (int dot, BAR heap, bandwidth, per-kernel times).
- `MISSION.md`: the brief the run followed. `p100-notes.md`: lessons from the earlier Tesla P100 run it builds on.
- `logs/`: `vulkaninfo` dump and `GGML_VK_PERF_LOGGER` per-kernel logs (start build and experiments) behind FACTS/RESULTS.
- `patches/`: reverted experiments worth revisiting (`git apply` on top of the branch). The int8 prefill GEMM in
  `exp4-mmq-ptq1_0-final.patch` is +34% pp but failed the KLD gate.

## Harness

Python 3 + PowerShell. Paths default to this checkout (`build/bin`) and the kit's `models/` next to the repo;
override with `BONSAI_KIT` / `BONSAI_MODEL_DIR`. Outputs (`runs/`, `slots/`, logits) are git-ignored.

```
python bench.py srv   --label NAME [--bin DIR] [--ctx 17408] [--tests tg0,tg16k,pp16k,hb]   # per-experiment protocol
python bench.py bench --label NAME --tests tg0,tg8k,tg16k,tg32k,pp0,pp16k                   # Step 0 llama-bench protocol
python gate.py kld-base        # reference logits from build/bin-5500m-start (f16 KV), ~3 GB
python gate.py kld|prompts|slot --label NAME [--bin DIR]
```

`srv` builds the 16K/32K slot files on first use (a long prefill), then restores them for every run.
`corpus-files.txt` lists the Markdown files (relative to the kit root) concatenated into the benchmark text.
