# Step 0 facts: Radeon Pro 5500M 8 GB, Bonsai 2 27B PTQ1_0, Vulkan (start of tuning, 2026-09-26)

Build under test: `5500m-start` tag (= main 871c6e476: PrismML prism 9a9394a89 + PR #188 + PR #187 + kit),
binaries in `llama.cpp/build/bin-5500m-start`. Machine: MacBook Pro 16" (2019) under Boot Camp, Windows 11,
i9-9980HK, on AC power. **The 5500M is the only GPU Windows sees and it drives the 3072x1920 panel**, so
the desktop and open apps (dwm, Chrome, VS Code, Steam) hold 1.0-1.35 GB of VRAM at idle. I did not
close any of the user's apps; every number below is with that background load (recorded per run).

## 1. GPU identity (vulkaninfo, full dump in `vulkaninfo.txt`)
| property | value |
|---|---|
| device | AMD Radeon Pro 5500M (Navi 14, RDNA1), deviceID 0x7340 |
| driver | AMD proprietary 24.10.1 (driverVersion 2.0.317), Vulkan 1.3.292 |
| subgroup size | default 64; `VK_EXT_subgroup_size_control`: min 32, max 64, required size allowed for compute -> **wave32 and wave64 both selectable** |
| max shared memory | 32768 bytes per workgroup |
| memory heaps | heap 0: 7.73 GiB DEVICE_LOCAL (budget 6.95 GiB at idle); heap 1: 15.7 GiB system; heap 2: **256 MiB DEVICE_LOCAL + HOST_VISIBLE (small BAR, no ReBAR), budget 230 MiB** |
| cooperative matrix | extension listed, not used by ggml (RDNA1 has no matrix cores; ggml reports "matrix cores: none") |
| fp16 | yes; bf16 no |

## 2. Integer dot product: HARDWARE ACCELERATED
`VK_KHR_shader_integer_dot_product`: integerDotProduct4x8BitPackedSignedAccelerated = **true**,
...UnsignedAccelerated = true, ...MixedSignednessAccelerated = false (8-bit signed/unsigned also true).
So the current PTQ1_0 kernel (`dotPacked4x8AccSatEXT` on signed data) runs on real `v_dot4_i32_i8`. The
P100's packed-FP16 decode tricks (notes 2A) are **not** the top priority here; the kernel is not ALU-bound
on emulated dots (see 5: the big mat-vecs already run at ~118-132 GB/s).

## 3. Measured memory bandwidth (llama.cpp `test-backend-ops perf`, Vulkan0)
| test | GB/s |
|---|---|
| ADD f32, 4096x512 with broadcast src1 (read+write) | **165** (best observed) |
| CPY f32->f16 512x3072 | 147 |
| SUM_ROWS f32 8192x8192 (256 MB, pure streaming read) | **132** |
| CPY f32->f32 8192x512x2 contiguous | 125 |
| PTQ1_0 x q8_1 mat-vec 17408x5120 (19.5 MB weights) | 106 (isolated), 125-132 inside the model |
Spec sheet: 192 GB/s (12 Gbps GDDR6, 128-bit). Usable peak is ~165 GB/s (86%), streaming read ~132.

## 4. Current performance, real config (q8_0 K / q4_0 V, fa on, ngl 99), median of 3, 60 s cooldown
llama-bench (`-r 1` per run, separate processes), depth via `-d`:
| test | tok/s (median) | reps | peak VRAM, llama process: dedicated + shared (MB) |
|---|---|---|---|
| tg128 @ depth 0 | **9.30** | 9.29 / 9.30 / 9.35 | 5573 + 494 |
| tg128 @ 8K | **9.68** | 9.54 / 9.68 / 9.68 | 6171 + 637 |
| tg128 @ 16K | **7.82** | 7.82 / 7.80 / 7.91 | 6436 + 580 |
| tg128 @ 32K | **does not start**: `vk::Device::allocateMemory: ErrorOutOfDeviceMemory` allocating the 550 MB K cache (llama-bench context 32768+128) | 3 x fail | - |
| pp512 @ depth 0 | **34.6** | 34.59 / 33.63 / 34.83 | 5980 + 509 |
| pp512 @ 16K | **15.8** | 16.96 / 15.84 / 14.06 | 6430 + 525 |

Prefill is very slow: 8K of context takes ~280 s, 16K ~750-860 s to prefill.

**Heartbeat** (llama-server, `-c 32768` started OK, q8_0/q4_0, -np 1, mmproj on CPU; 16K prefix cached, then
300 new tokens + 64 generated, greedy): **TTFT 15.94 s, total 23.69 s** (median; reps 15.74/15.94/16.07 s TTFT,
23.58/23.69/23.78 s total). The 300-token append runs at ~19 t/s, decode at ~8.1 t/s.
Server peak VRAM: process 6487 MB dedicated + 596 MB shared; adapter 7877 MB dedicated + 804 MB shared.

**Achieved decode bandwidth.** Bytes per token = GPU weights 5395 MiB (5.66 GB; token_embd stays on CPU)
+ recurrent state read+write (2 x 144 MiB = 0.30 GB) + KV read at depth (26,624 B per cached token:
16 layers x 4 KV heads x 256 x (34/32 + 18/32) B).
| depth | bytes/token | tok/s | GB/s | % of 165 GB/s peak |
|---|---|---|---|---|
| 0 | 5.96 GB | 9.30 | 55 | **34%** |
| 16K | 6.40 GB | 7.82 | 50 | **30%** |

**VRAM at the real config** (llama-server log): model 5395.33 MiB, KV 832 MiB (K 544 + V 288 at 32K),
recurrent state 149.62 MiB, compute buffer 166.28 MiB (+52 MiB host) = 6543 MiB before driver/runtime
overhead; plus a lazily grown f16 K/V scratch for prefill at depth (4 KB per cached token: 64 MiB at 16K,
128 MiB at 32K, never freed). Device total 8176 MiB, ~1.1-1.35 GB taken by the desktop -> 32K does
not fit with >= 400 MB headroom on this desktop; llama-bench at 32K fails outright, llama-server at 32K
starts but ~600 MB of the process sits in shared (system) memory.

## 5. Per-kernel time for one decode token (`GGML_VK_PERF_LOGGER`, depth ~256; logs `perflog-start.txt`)
Start build: **112.8 ms** GPU time per token, **~2037 dispatches** (ops + fused ops) per token.
| op | count/token | us each | ms/token |
|---|---|---|---|
| CPY (recurrent state write-back, conv state) | 96 | 251 | **24.1** |
| GET_ROWS (recurrent state gather) | 97 | 243 | **23.6** |
| MUL_MAT_VEC ptq1_0 17408x5120 (ffn gate/up) | 128 | 150 | 19.2 |
| MUL_MAT_ADD ptq1_0 5120x17408 (ffn down + residual) | 64 | 147 | 9.4 |
| MUL_MAT_VEC ptq1_0 10240x5120 (GDN qkv) | 48 | 94 | 4.5 |
| MUL_FWHT (Hadamard sign MUL + FWHT) | 210 | 20.8 | 4.4 |
| FLASH_ATTN_EXT (256 cells) | 16 | 193 | 3.1 |
| MUL_MAT_VEC ptq1_0 5120x6144 / 6144x5120 (ssm_out, z) | 96 | 60 | 5.9 |
| GATED_DELTA_NET | 48 | 54 | 2.6 |
| output head ptq1_0 248320x5120 | 1 | 2129 | 2.1 |
| everything else (~1400 small ops: norms, sigmoid, softplus, concat, rope, ...) | | 5-20 | ~13 |

**Root cause of the slowdown vs the earlier ~14.5 tok/s:** ggml-vulkan places device buffers in
DEVICE_LOCAL|HOST_VISIBLE memory first when it exists. On this laptop that is the 256 MiB BAR window,
shared with the desktop; the recurrent-state (150 MiB) and compute buffers land there and are demoted
to system memory, so the state GET_ROWS/CPY run at ~12 GB/s (PCIe). With
`GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1` the same ops take 29 us / 25 us, the token takes 71.5 ms
and tg128 is **14.32 tok/s**. That is experiment 1.

With that fixed (71.5 ms/token), the weight mat-vecs take ~45 ms at ~125 GB/s (close to the ~132 GB/s
streaming-read ceiling), and ~26 ms goes to everything else: FWHT 4.5, flash attention 3.2, state
gather/copy 5.2, GDN 2.6, norms 2.4, and ~1500 tiny element-wise dispatches at 5-7 us each.

**CPU time per decode token:** 27.6 ms (process CPU time, tg144 minus tg16, /128) at 106 ms/token wall,
so CPU submission is not the limiter now. It could become one below ~30 ms/token.

## Other facts found
- The CPU backend runs PTQ1_0 at ~0.3 tok/s (32 layers on CPU), so a CPU KLD reference is infeasible.
  The KLD reference is the start build with f16 KV at 1K context (see gate.py).
- The fork already has two GDN optimizations that are disabled on Vulkan: "rows mode" (GDN reads the state
  row in place, no gather) and "raw gates" (sigmoid/softplus inside GDN). Both are experiment candidates.
- Default `n_rs_seq = 0` in llama-server, so decode takes the gather (GET_ROWS) + copy (CPY) state path.
