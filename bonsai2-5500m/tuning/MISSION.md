# Mission: tune Ternary Bonsai 2 27B (PTQ1_0) on a Radeon Pro 5500M 8GB, Vulkan, Windows

You are running unattended. Tyler may be away. Do not stop to ask questions: decide, log the
decision, and continue. First read `bonsai2-optimization-notes.md` (lessons from a successful
tuning run of the same model on a Tesla P100) and the current state described below.

## Current state (already done; do not redo)
- This repo is PrismML's llama.cpp fork (upstream/prism at 9a9394a89) plus open upstream PRs
  #188 and #187, merged on main:
  - #188: dedicated q8_1 mat-vec shader `mul_mat_vecq_ptq1_0.comp` (2 lanes per 128-element
    block, 4 rows per workgroup, trits decoded 2 bytes at a time into `dotPacked4x8AccSat`,
    ternary -1 offset folded into q8_1 block sums), multi-column path, fixed MUL_MAT_ID q8_1
    pipeline selection, Vulkan PQ2_0 support.
  - #187: 64-entry trit lookup table, vectorized `dequantize4`, MUL (Hadamard signs) fused into FWHT.
- Result so far: decode ~1.6 -> ~14.5 tok/s on this GPU.
- Launcher: `bonsai2-5500m/start-server.ps1`. Config: PTQ1_0, 32K context, K cache q8_0, V
  cache q4_0, `-np 1`, `--no-mmproj-offload`, `-fa on`. q8_0/q8_0 at 32K fits but drops to
  ~9 tok/s (VRAM nearly full). q8/q4 keeps ~14 tok/s; perplexity vs f16 KV: -0.07% at 2K, +0.36% at 16K.
- Unpatched baseline build: `build/bin-b10687-baseline`.

## Step 0: establish facts before changing anything
Write the answers into `bench/FACTS.md`:
1. GPU identity via `vulkaninfo`: device name, driver version, subgroup size(s) and whether
   `VK_EXT_subgroup_size_control` allows wave32 and wave64, max shared memory, and memory heaps.
2. **Is int8 dot product hardware-accelerated?** Check `VK_KHR_shader_integer_dot_product`
   properties, especially `integerDotProduct4x8BitPackedSignedAccelerated` /
   `...MixedSignednessAccelerated`. If NOT accelerated, the current kernel is compute-bound
   on emulated dot products, exactly like the P100, and the P100's packed-FP16 decode tricks
   (notes section 2A: subnormal masking, 0x6400 magic, packed FP16 FMA) become the top priority.
3. Measured memory bandwidth (write a small Vulkan copy/read benchmark, or use llama.cpp's
   tooling). Do not trust spec sheets.
4. Current performance with the real config (q8_0 K / q4_0 V, 32K context), median of 3 runs
   with a 60 s cooldown between runs:
   - tg128 at depth 0, 8K, 16K, and 32K (llama-bench `-d`)
   - pp512 at depth 0 and 16K
   - **Heartbeat latency:** append a 300-token message to an already-cached 16K prefix and
     generate 64 tokens. Report time to first token and total time.
   - Achieved decode bandwidth = bytes read per token / time per token, as a % of measured peak.
   - Peak VRAM use for each config.
5. Per-kernel time breakdown for one decode token (use `GGML_VK_PERF_LOGGER` or equivalent)
   and number of dispatches per token. Note CPU time per token, too: on the P100, graph
   submission cost ~8 ms/token of CPU time.

## Objective, in priority order
1. Decode speed at 16K and 32K depth with the q8_0/q4_0 KV cache (the real workload).
2. Heartbeat latency (300-token append to a cached prefix + 64-token reply).
3. pp at depth.
Do not regress depth-0 tg by more than 2%.

## Hard constraints
- **VRAM headroom:** peak usage must stay at least ~400 MB below total VRAM at 32K context.
  If usage approaches full, the Windows driver spills to system RAM and speed collapses.
  Report peak VRAM for every win.
- Must work with the q8_0 K / q4_0 V cache at 32K. Other KV types can be explored, but the
  final config must fit at 32K with headroom.
- One sequence (`-np 1`) is the target. Optimizations may assume a single sequence, but must
  be gated so multi-slot use still works (fall back, don't break).

## Correctness gate (every win must pass all of these)
1. `test-backend-ops` for every op you touch (compares the Vulkan backend against the CPU
   backend). Add tests for any new fused op.
2. KL divergence against a reference: compute reference logits once with
   `llama-perplexity --kl-divergence-base` using the most accurate available config (CPU
   backend if it supports PTQ1_0 and fits in RAM; otherwise the current build with f16 KV at
   short context). A change must not make mean KLD worse than the current build by more than
   noise. Do NOT require bit-identical output to stock; different summation orders are fine.
3. A fixed set of 8 prompts (math, code, translation, facts) must reach correct final answers
   with enough max_tokens for thinking to finish.
4. Slot save/restore (`--slot-save-path`, save, restart, restore, greedy continue) must be
   byte-identical to a continuous run of the same build.

## Ideas to try (from the P100 run; see the notes file for details)
- If int8 dot is emulated: packed-FP16 decode path for PTQ1_0 with exact integer partials.
- Workgroup geometry sweeps per matrix shape: rows per workgroup, lanes per block, wave32 vs
  wave64 via subgroup size control. On the P100 the best geometry was never the default.
- If a load-only variant of the kernel is as slow as the full kernel: stage through shared
  memory with wide (128-bit) coalesced loads.
- Launch/dispatch reduction (barriers between dispatches are costly on AMD):
  - FWHT kernel also emits the q8_1 activations the mat-vec needs (skip the separate quantize).
  - Skip GET_ROWS of the recurrent GDN state for a single sequence; read the cache row directly.
  - Fuse the short-conv state chain (GET_ROWS + CONCAT + CPY + SSM_CONV + SILU) into one kernel.
  - Fuse ADD + RMS_NORM + MUL, and fuse FFN gate + up + SWIGLU into the mat-vec.
- Attention at depth: check that flash attention with q8_0 K / q4_0 V uses a fast path rather
  than dequantizing the whole cache; try splitting the KV dimension across workgroups
  (flash-decoding) so single-sequence decode keeps all CUs busy at 16K-32K.
- Prefill: dedicated PTQ1_0 -> f16 dequant, and check the GEMM path's achieved TFLOPS vs peak.
- Cheap re-tests of things that failed on the P100 (notes section 2D); different hardware,
  different answers.

## Method
- Work on a new git branch `5500m-tuning` from current main. Tag the starting point
  `5500m-start` before the first change.
- One change per experiment. A win must beat noise by at least 2% on the target metric;
  re-run anything close. Commit wins with the metric deltas in the message. Revert losses
  completely.
- Laptop GPU: plug in power, wait 60 s between timed runs, report the median of 3.
- Log EVERY experiment (wins, losses, failures) as a row in `bench/RESULTS.md`:
  | time | commit/label | change | tg@0 | tg@16K | tg@32K | heartbeat | pp@16K | VRAM peak | correctness | verdict | note |
  Read RESULTS.md before each experiment so nothing is retried.
- After each experiment, print its new RESULTS.md row in the conversation.

## Done when
6 consecutive experiments produce no win, or all reasonable ideas are exhausted. Then write
`bench/SUMMARY.md`: starting vs final numbers for every metric, each winning change and WHY it
helped, notable failures, peak VRAM, the exact recommended `start-server.ps1` settings, and
remaining ideas. Print it in full.
