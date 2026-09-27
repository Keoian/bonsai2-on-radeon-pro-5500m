# Bonsai 2 27B (PTQ1_0) on the Radeon Pro 5500M 8 GB: tuning summary

Branch `5500m-tuning` in `C:\git\bonsai2\llama.cpp`, 7 commits on top of tag `5500m-start` (not pushed).
Full log of all 15 experiments: `RESULTS.md`. Step 0 facts: `FACTS.md` (both in this folder).
Stopped after 6 consecutive experiments without a win (exp 9, 11, 12, 13, 14, 15).

## Start vs final

Same session, back to back, same protocol (llama-server, q8_0 K / q4_0 V, `-np 1`, `-fa on`, `-c 17408`,
16K depth from a restored slot, median of 3, 60 s cooldowns):

| metric | start (`5500m-start`) | final (`bb9f8fda0`) | change |
|---|---|---|---|
| decode tg @ depth 0 | 9.16 tok/s | **16.74 tok/s** | **+83%** |
| decode tg @ 16K | 8.24 tok/s | **13.90 tok/s** | **+69%** |
| heartbeat (300-token append to cached 16K + 64 tokens): TTFT | 15.83 s | **9.67 s** | **-39%** |
| heartbeat: total | 23.57 s | **14.33 s** | **-39%** |
| prefill pp512 @ 16K | 19.55 tok/s | **34.90 tok/s** | **+79%** |
| peak VRAM, llama-server process (dedicated + shared) | 6081 + 581 MB | 6197 + 431 MB | +116 / -150 MB (buffers moved out of the BAR window) |
| KLD vs f16-KV reference (12 x 1K chunks) | 0.002896 | 0.002896 | identical |

llama-bench, Step 0 protocol (median of 3, 60 s cooldown):

| test | start | final |
|---|---|---|
| tg128 @ depth 0 | 9.30 | **16.89 (+82%)** |
| pp512 @ depth 0 | 34.6 | **49.7 (+44%)** |
| tg128 @ 8K / 16K | 9.68 / 7.82 | could not run: out of device memory at context creation (see VRAM below) |
| tg128 @ 32K, pp512 @ 16K | OOM / 15.8 | out of device memory |

Other Step 0 numbers, start -> final:
- **GPU time per decode token** (perf logger, shallow depth): 112.8 ms -> 59.6 ms. Dispatches per token: 2037 -> 1941.
- **Achieved decode bandwidth** (5.96 GB per token: weights plus recurrent state): 55 GB/s (34% of the 165 GB/s measured peak) -> ~100 GB/s (61%). The weight mat-vecs alone now run at ~126 GB/s, about 95% of the 132 GB/s streaming-read ceiling.
- **Flash attention per layer** (q8_0/q4_0, kv 16K): prefill of 512 tokens 666 -> 224 ms; decode 863 -> 768 us.
- **CPU time per decode token:** 27.6 ms -> 24.4 ms (at ~60 ms wall), not the limiter.

**tg @ 32K was never measurable with the q8_0/q4_0 cache at 32K, before or after.** The reason is the desktop,
covered next.

## The environment matters more than any kernel

The 5500M is the only GPU Windows sees (Boot Camp) and it drives the 3072x1920 panel. dwm plus open apps held
1.0-1.4 GB of VRAM during the run: Chrome, VS Code, Steam, and later a Remote Desktop client and the Snipping
Tool. The adapter's usable dedicated memory tops out near 7.8 GB, which leaves llama about 6.4-6.8 GB.
- The real 32K q8_0/q4_0 config needs ~6.9 GB: model 5395 MiB + KV 832 MiB + recurrent state 150 MiB + compute 166 MiB.
- 32K therefore cannot be fully resident, and the 400 MB headroom constraint cannot be met at 32K on this desktop.
- A 32K server either fails to allocate, or starts with ~500 MB of hot buffers in system RAM and decodes at ~6.5 tok/s.
- Even 16K (`-c 17408`) sat at the edge late in the run. Servers occasionally failed on 39 KB allocations, and
  llama-bench depth runs no longer fit.

I did not close any apps. **To use 32K, close or GPU-disable Chrome, VS Code, Steam and similar first. Otherwise use 16K.**

## Winning changes and why they helped

1. **Avoid host-visible VRAM when the BAR heap is small** (`37f35dd53`; tg +63%/+46%).
   - ggml-vulkan put every device buffer in DEVICE_LOCAL|HOST_VISIBLE memory first.
   - Without resizable BAR that is a 256 MiB window shared with the desktop (budget 230 MiB).
   - The recurrent state and compute buffers landed there and were demoted to system memory, so the per-token
     state GET_ROWS/CPY ran over PCIe at ~12 GB/s (48 ms/token).
   - Only use that heap when it is at least 1 GiB. This was the main cause of the 14.5 -> 9 tok/s regression the user had seen.
2. **GDN reads the recurrent state row in place** (`6771079d3`; tg@16K +8%, heartbeat total -9%).
   - Ported the fork's "rows mode" (CPU/Metal only) to Vulkan and used it on the default n_rs_seq = 0 path, gated
     to one sequence with no extra state relocation.
   - This removes a 3 MB gather per GDN layer (48 per token).
3. **Fuse the GDN state write-back into the GDN kernel** (`a751c6df8`; tg +3-4%, pp +7%).
   - GDN writes its final state directly into the cache row, which removes a 3 MB copy per layer.
   - In place is safe because each invocation reads its own elements before writing them. New fusion test added.
4. **Dword-at-a-time PTQ1_0 decode in the prefill GEMM loader** (`13b1079f4`; pp +3.7%, heartbeat TTFT -8%).
   - The float mat-mat decoded every trit through a branchy per-element function. 15 of 16 groups are two aligned
     dwords at one trit level, so the mat-vec's 16-bit-lane multiply trick applies.
   - Same values, full precision. GEMM 1.99 -> 2.36 TFLOPS.
5. **Shared-memory FWHT for 1024-wide transforms on AMD** (`dda7aad3a`; tg +5%).
   - The one-subgroup-per-row wave64 variant left the GPU nearly idle for decode's 5-17 rows.
   - The 256-thread shared-memory variant cut each of the ~306 transforms per token from 21.3 to 7.7 us.
6. **Flash-attention tuning for head size 256 on RDNA** (`652b83fda`; pp@16K +64%, heartbeat TTFT -33%, tg@16K +5%).
   - 16 rows per workgroup spilled registers in the scalar FA shader; 8 is 3x faster for prefill at depth.
   - d_split 16 speeds up few-row decode. Added Bonsai-shaped FA perf cases.
7. **64x128 PTQ1_0 mat-mat tile on RDNA1, chosen by column padding** (`bb9f8fda0`; pp +6.6%).
   - The large tile is disabled on AMD's driver, so prefill ran on 64x64. The 64x128 tile is ~10% faster per column.
   - It is used only where its padding is small: the server splits a 512-token append into 508 + 4, which gets
     the wide tile; the 296-token heartbeat keeps 64x64.

All seven pass every gate:
- `test-backend-ops` for the touched ops: GDN 49/49 incl. rows mode, the new GDN_STATE_CPY and mat-mat cases, FA 5150/5150, Hadamard 27/27.
- KLD identical to the start build.
- 8/8 fixed prompts correct.
- Slot save -> restart -> restore -> continue is byte-identical to a continuous run (only the 40 new tokens are evaluated after restore).

## Notable failures (details in RESULTS.md)

- **PTQ1_0 integer-dot GEMM (MMQ) for prefill:** pp +34%, heartbeat TTFT -23%, GEMM 3.5 TFLOPS.
  - Failed the KLD gate: 0.00433 vs 0.00290, from 8-bit prefill activations. Reverted; patch kept in `patches/exp4-mmq-ptq1_0-final.patch`.
  - This is the largest remaining prefill opportunity if a more accurate activation format can be found.
- **Mat-vec geometry sweep** (rows 2-16, wave32 vs wave64, workgroup size): the existing 4 rows / wave64 is best. The kernel is at the bandwidth roofline.
- **GDN raw gates on Vulkan** (-192 dispatches/token): ~1 ms of GPU time saved, not visible end to end. Near-miss.
- **FWHT writes the q8_1 copy for its mat-vec consumer, combined with raw gates:** +0.7% at 16K.
  - The graph optimizer runs several FWHTs before their consumers, so the single staging buffer is overwritten first.
  - A probe that skips all quantize dispatches bounds the idea at +5%.
- **Skipping the f16 K/V dequant scratch** (VRAM): the real model's KV views never use that path, so there was nothing to save.
- **No graph reordering; mask optimization for decode FA; medium-tile sweep at heartbeat size:** all within noise or worse.
- **Memory priority** (`GGML_VK_ENABLE_MEMORY_PRIORITY`) does not keep llama resident against the desktop.

## Recommended `start-server.ps1` settings

All seven wins are code defaults, auto-gated to this GPU (RDNA1 / small BAR), so no new flags or environment
variables are needed. Rebuild `build/bin` from `5500m-tuning`. It is already built there; the kit's `build/bin` now holds these binaries.

```
.\start-server.ps1
    # = -Model ptq1 -Ctx 16384 -CacheK q8_0 -CacheV q4_0 -Parallel 1 (-np 1), -fa on, mmproj on CPU (--no-mmproj-offload)
.\start-server.ps1 -Ctx 32768
    # only after closing GPU-heavy apps (Chrome, VS Code, Steam, Remote Desktop); needs ~6.9 GB free VRAM
```

- The 16K default is the user's own uncommitted edit in `bonsai2-5500m/start-server.ps1`. I left it uncommitted,
  and it matches what this run found.
- The committed launcher on `main` still defaults to 32K. Its README claim ("32k fits in 8 GB") holds only on a
  near-empty desktop.
- For per-request thinking control use `thinking_budget_tokens` (P100 notes); it keeps the cached prefix intact.

## Remaining ideas

1. **Decode flash attention at depth:** ~12 ms/token at 16K for 27 MB of KV, ~4x off bandwidth. Tile and split-K
   tuning is exhausted. A dedicated GQA-6 decode kernel (whole KV rows per thread, no 16-lane reductions per 32 positions) is the next step. It is worth ~+15% tg@16K.
2. **Accurate MMQ prefill:** the int8 GEMM is 1.35x faster than the float path but loses KLD. Options:
   - two-term (hi/lo) int8 activations;
   - or MMQ only for the GEMMs whose inputs quantize well.
3. **FWHT -> q8_1 with per-tensor staging:** instead of one shared `prealloc_y`, with graph-aware invalidation. Bound: +5% decode.
4. **Dispatch count:**
   - ~1940 ops per token at a 5-6 us floor.
   - Fuse ADD + RMS_NORM + MUL + sign MUL into the FWHT (P100 exp37).
   - Revisit raw gates together with other fusions.
5. **VRAM:**
   - Anything that trims the resident set helps more on this laptop than kernel work. Examples: `-ub 256` (smaller
     compute buffer, prefill cost to be measured), or a smaller V cache type if it passes KLD.
   - The practical fix is fewer GPU-hungry apps while the model runs.

The reverted experiments worth revisiting are in `patches/` (exp4, exp9, exp12, warptile sweep hooks); the rest
are described in RESULTS.md. Binaries for each step were kept locally in `build/bin-*`
(start, exp2, exp3, exp4, exp5, exp7, exp8, exp9, exp10, exp11, exp12, exp14).
