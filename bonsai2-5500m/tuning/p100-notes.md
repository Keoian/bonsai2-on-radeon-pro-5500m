# Ternary Bonsai 2 27B: optimization notes from the Tesla P100 run, and a plan for the Radeon Pro 5500M

This file summarizes what Claude Code changed in PrismML's llama.cpp fork to speed up
Ternary Bonsai 2 27B on a Tesla P100 (CUDA, sm_60), what did and didn't work, and how those
lessons might carry over to a Radeon Pro 5500M 8GB.

**Source of truth:** the private repo `Keoian/llama.cpp-p100`, branch `p100-tuning`, tag
`phase1-final`. Every winning commit message carries its before/after numbers. The full
experiment log (wins, losses, and failures) is `/mnt/user/llm-dev/bench/RESULTS.md` on the
Unraid server, backed up to `/mnt/user/backups/llm-dev-phase1/`.

---

## 1. Results on the P100

Model: `Ternary-Bonsai-2-27B-PQ2_0.gguf`, one P100 at a 180 W power limit, flash attention on.

| Metric | Stock fork (`adfffbe41`) | Tuned (`phase1-final`, `0139eb3f5`) | Change |
|---|---|---|---|
| Generation, tg128 | 18.26 tok/s | 47.23 tok/s | 2.6x |
| Prompt, pp512 | 130.4 tok/s | 237.7 tok/s | +82% |
| Prompt, pp4096 | 129.0 tok/s | 227.3 tok/s | +76% |

Validation so far (phase 2, still running when this was written):
- Slot state save/restore and prefix-cache reuse: byte-identical to stock. **Pass.**
- Perplexity: fine. Top-1 agreement with stock is 96.58%, below the original 99% gate. The
  cause is the new cuBLAS GEMM algorithms. Against an fp32 reference, though, the tuned build
  is about 2x closer than stock (KLD 0.00187 vs 0.00413). Generation-path KLD passes at 99.33%
  top-1. Verdict: the tuned build is more accurate, not less.
- Found and fixed a stock bug where `--reasoning-budget 0` did not stop thinking (commit
  `0a4b14973`). `thinking_budget_tokens` is the only per-request thinking control that keeps
  a cached prompt prefix intact; `enable_thinking` and `reasoning_effort` rewrite the system block.
- Stock at 32K depth: pp512 98.7, tg128 16.1 tok/s. Tuned long-context numbers were pending.

---

## 2. What changed, grouped by idea

Experiment numbers (expN) refer to rows in RESULTS.md.

### A. Token generation: the mat-vec kernel (the biggest win, about 18 -> 39 tok/s)

On sm_60 there is no DP4A (int8 dot product) instruction. The stock quantized mat-vec path
(MMVQ) emulated DP4A, so generation was **compute-bound** at about 122 GB/s out of the card's
732 GB/s.

- **exp1: native packed-FP16 decode.** `vec_dot_pq2_0_q8_1` on sm_60 unpacks 2-bit codes with
  byte-permute and bit ops, then uses packed half2 FMA (HFMA2) instead of emulated DP4A. The
  integer partial sums stay exact. +11% tg.
- **exp2 / exp3: launch geometry.** For K=5120 shapes, rows were block-overhead bound. Forcing
  a small-K geometry, then a dedicated Pascal parameter table (2 warps x 4 rows), gave +48% and
  then +4%.
- **exp5: dedicated Pascal PQ2_0 mat-vec kernel.** 2-bit codes are masked straight into FP16
  subnormals so each weight pair costs 1 logic op + 1 HFMA2, with exact integer partials and
  constant-stride row pointers. The inner loop dropped from 3.9 to 2.7 instructions per weight.
  +6.7%.
- **exp8: shared-memory staging.** A diagnostic showed a load-only kernel was as slow as the
  full kernel: it was bound by the L1/texture pipe servicing many 2-byte loads. Each warp now
  stages 8-block row groups plus 32 q8_1 activation blocks into shared memory with coalesced
  128-bit loads. +15.5%.
- **exp9: 1 warp x 8 rows**, with no cross-warp split of 1024-column groups and no shared-memory
  reduction. +3%.

### B. Prompt processing: dequantize, then a fast fp16 GEMM

Prompt processing on sm_60 goes through dequantize -> cuBLAS, because the fast quantized
matmul (MMQ) needs DP4A.

- **exp12: dedicated PQ2_0 -> f16 dequant kernel.** 16 elements per thread, uses the 0x6400
  "magic number" trick (build the fp16 bit pattern, then HFMA2 + HMUL2 to scale), exact,
  128-bit stores. The generic dequant ran at about 105 GB/s. +9% pp.
- **exp13: pick the right cuBLAS algorithm.** cuBLAS's default heuristic chose a Maxwell-era
  kernel running at about 8 TFLOPS. A standalone sweep of GEMM algorithms across all six model
  shapes found `CUBLAS_GEMM_ALGO2` at 15-16 TFLOPS. **+47% pp from a tiny change.**
- **exp14:** bf16 weights on pre-Ampere GPUs use exact bf16->f32 conversion plus SGEMM instead
  of a slow bf16 kernel. +2.5%.
- **exp34:** for GEMMs with M > K (FFN up/gate, QKV, gates), swap operands (C^T = B^T A) with
  `CUBLAS_GEMM_ALGO5`, and fold the transpose back into the f16->f32 output conversion. +5%.
- **exp36:** gated delta net (GDN) prefill: raw gates (sigmoid beta, softplus/exp g) computed
  once per token and head instead of per column, plus 4 columns per warp so q/k registers are
  reused. Each half was below the noise bar alone; together, +4.5%.

### C. Fewer, fatter kernel launches (the decode long tail, about 39 -> 47 tok/s)

Bonsai 2 is a hybrid model: about 75% linear-attention (gated delta net + short conv) layers
and 25% full attention, with Hadamard (FWHT) rotations in front of the ternary matmuls. That
makes for many small kernels per token.

- **exp22:** the FWHT kernel also writes the q8_1-quantized copy of its output, so
  `quantize_q8_1` launches went from 401 to 0 per token. +3.6%.
- **exp24:** skip GET_ROWS of the recurrent state (3 MB per layer) when it only feeds GDN with a
  single sequence; GDN reads the cache row directly. +3.2%. **Assumes one sequence.**
- **exp25:** enable the existing ADD + RMS_NORM + MUL fusion on Pascal, accepting in-place
  aliasing only when every overlap is an exact alias. +2.2%.
- **exp26:** FFN gate + up + SWIGLU fused into the mat-vec kernel (4 up rows + 4 gate rows
  per warp). 192 launches per token became 64. +2.7%.
- **exp30:** the short-conv state chain (GET_ROWS + CONCAT + CPY + SSM_CONV + SILU) became one
  kernel that updates the state in place. 4 launches per layer became 1. +2.8%.
- **exp37:** FWHT writes pre-paired, pre-scaled half2 ("a16") activations so the mat-vec loads
  them directly, and the residual ADD + RMS_NORM + MUL is fused into the FWHT. Each half was
  below the bar alone (exp29, exp32); together, +3.3%. The first attempt had a scale bug that
  produced gibberish, which the correctness gate caught.

### D. What did NOT help on the P100 (cheap to re-test elsewhere; results may differ)

- CUDA graphs (exp4, exp41): CPU submission wasn't the bottleneck.
- `__launch_bounds__` for higher occupancy (exp7): occupancy wasn't the limiter.
- Register double-buffering (exp10), persistent kernels (exp39), multi-warp shared activation
  stages (exp20), splitting gate/up across warps (exp42).
- Changing thread counts for single-row RMS norm (exp11, exp23, exp31): latency bound.
- cuBLAS with fp32 output (exp33): the P100 ran it at fp32 rate.
- Per-shape cuBLAS algorithm choice (exp15): below run-to-run noise.
- Non-default `-ub` micro-batch sizes (exp16, exp45): 512 stays best.
- **PTQ1_0 packing (exp18): 8.6 tok/s vs about 39 for PQ2_0.** PTQ1_0 has no dedicated kernel,
  so it uses the generic path with emulated DP4A and a base-3 decode. It's the smaller file
  (1.76 vs 2.16 bits per weight), but it costs about 4 ALU ops per weight pair versus 2. On a
  compute-bound card it loses; on a bandwidth-bound card with a good kernel it might win.
  **This matters a lot for the 5500M; see below.**

---

## 3. Transferable lessons (backend-agnostic)

1. **Measure achieved GB/s against peak before optimizing.** Stock on the P100 reached 17% of
   bandwidth, which proved generation was compute- or issue-bound, not memory-bound. That
   single number told Claude where to look.
2. **Write a dedicated mat-vec kernel for the packed format.** Generic quantized paths assume
   int8 dot-product hardware. On GPUs without it, decoding into packed FP16 with bit tricks
   (subnormal masking, the 0x6400 magic number) and using packed FP16 FMA was the winning
   pattern.
3. **When a load-only kernel is as slow as the full kernel, fix loads, not math.** Stage
   through shared/local memory with wide (128-bit) coalesced loads.
4. **Sweep launch geometry per shape.** The best rows-per-group and groups-per-block combination
   was not the default, and it moved every time the kernel changed.
5. **For prefill, check the GEMM library's choice.** The default heuristic can be far from the
   best algorithm on older hardware.
6. **On hybrid models, count launches per token.** Fusing the small linear-attention and norm
   kernels added up to about 20% of decode speed.
7. **Combine near-misses.** Several changes were +1.5% alone but +3-4% together.
8. **Gate correctness properly:** KLD against an **fp32 reference** (not just "matches stock"),
   plus byte-identical slot save/restore and prefix reuse, plus a set of fixed prompts checked
   for correct final answers (allow enough max_tokens for the model to finish thinking).

---

## 4. Plan for the Radeon Pro 5500M 8GB

### Hardware facts to verify first

From memory, **confirm these with real tooling before relying on them**:
- Navi 14, RDNA1, likely gfx1012. 24 CUs, wave32 native (wave64 capable).
- 8 GB GDDR6 on a 128-bit bus, roughly **192 GB/s**. That is about a quarter of the P100's
  bandwidth, so this card will almost certainly be **bandwidth-bound**, the opposite of the P100.
- Packed FP16 (`v_pk_fma_f16`) at roughly 2x fp32 rate. The P100's FP16 tricks should carry over.
- **Check whether this chip has int8 dot-product instructions (`v_dot4_i32_i8`).** Some RDNA1
  variants do and some don't. If it does, the generic int8 path may already be decent and
  the kernel strategy changes.
- Laptop GPU: it thermal-throttles. Benchmark with cooldowns, plugged in, and record temps.

### Which backend

- **ROCm/HIP:** RDNA1 is not officially supported, and it isn't available on macOS. Probably a dead end.
- **Metal (macOS):** the PrismML fork has Metal kernels, but they're tuned for Apple Silicon.
  AMD GPUs on Intel Macs lack some Apple-GPU features, so check which kernels actually run and
  how fast before tuning.
- **Vulkan (Windows via Boot Camp, or Linux):** probably the best-supported path for RDNA1 in
  llama.cpp. Confirm the fork's Vulkan backend implements the Hadamard/FWHT runtime transform.
  Without it, output is garbage rather than an error.
- **Step zero:** get a correct stock build running on whichever backend works, and record the
  baseline and the backend used.

### Memory budget: 8 GB is tight

- PQ2_0 weights are 7.25 GB, which likely leaves too little room for KV cache, recurrent state,
  and compute buffers. **PTQ1_0 (5.93 GB) is probably the only viable packing.**
- Skip the vision tower (mmproj). Use `--parallel 1`.
- KV cache for the 16 full-attention layers is very roughly 60+ KB per token in FP16 (measure
  it). That means 16K context is realistic; 32K may need a q8_0 KV cache, which must pass the
  KLD gate.

### What to try, in order

1. **Baseline and a bandwidth check.** Measure tg128 and pp512 at short context and at 16K
   depth. Compute achieved GB/s for generation (bytes of weights read per token divided by
   time per token). Theoretical ceiling at 192 GB/s with a 5.9 GB model is roughly 30 tok/s;
   a realistic target is well below that.
2. **Dedicated PTQ1_0 mat-vec kernel.** On the P100, PTQ1_0 lost because it had no dedicated
   kernel. On a bandwidth-bound card, the 17.5% smaller file could win *if* the base-3 decode
   is cheap enough. Try lookup tables in shared/local memory, and decoding into packed FP16
   with the same bit tricks as exp5 and exp12.
3. **Launch geometry sweep** for wave32 vs wave64 and rows per workgroup, per matrix shape.
4. **Local-memory (LDS) staging with wide loads** if loads turn out to be the limiter (the
   exp8 pattern).
5. **Launch-count fusions** from section 2C. These are backend-independent ideas: q8 (or a16)
   activations emitted by the FWHT kernel, skipping the recurrent-state GET_ROWS for a single
   sequence, the fused short-conv state kernel, fused ADD+RMS_NORM+MUL, and fused
   gate+up+SWIGLU. Vulkan dispatch overhead may make these worth *more* than on CUDA.
6. **Prefill:** check what GEMM path the backend uses for f16, measure its TFLOPS against
   peak, and try a dedicated PTQ1_0->f16 dequant kernel (the exp12 pattern).
7. **Port the stock bug fix** for `--reasoning-budget 0` (commit `0a4b14973`) and use
   `thinking_budget_tokens` for per-request thinking control.
8. **Re-test the P100 losers cheaply** (section 2D). Different hardware, different answers.

### Method (same as the P100 run)

- One change per experiment. Commit wins with their numbers; revert losses completely.
- Log every attempt, including failures, so nothing gets retried.
- Correctness gate on every win: KLD against an fp32 reference must not get worse, fixed prompts
  must reach correct final answers, and state save/restore must stay byte-identical.
- A change must beat noise by 2% or more; re-run anything close.
- Stop after 6-8 consecutive experiments without a win, then write a summary with the final
  numbers, recommended runtime flags, and remaining risks.
