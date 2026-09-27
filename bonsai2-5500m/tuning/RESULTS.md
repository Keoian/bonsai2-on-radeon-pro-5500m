# 5500M tuning: experiment log

Model: Ternary-Bonsai-2-27B-PTQ1_0, Radeon Pro 5500M 8 GB, Vulkan (AMD proprietary 24.10.1), Windows 11.
Real config: 32K context, K cache q8_0, V cache q4_0, `-np 1`, `-fa on`.

**Protocol decision (14:15).** On this desktop (dGPU drives the 3K panel; dwm + Chrome + VS Code + Steam hold
1.2-1.35 GB VRAM, and the adapter's usable dedicated memory tops out near 7.8 GB) the real 32K q8_0/q4_0
config (~6.9 GB for llama) cannot be fully resident: the late, hot allocations (recurrent state, compute
buffer) land in system memory and decode collapses to ~6.5 tok/s regardless of kernel work. So the
per-experiment metrics use `-c 17408` (16K depth + heartbeat room, same q8_0/q4_0 KV), which is resident,
and tg@32K is a secondary metric taken on a `-c 32768` server for milestone builds. Making 32K fit is its
own experiment track. I did not close any of the user's apps.

Metric protocol ("srv", used for every experiment): one llama-server with the real KV config; depth comes from
restoring a saved 16K / 32K-token slot and appending 16 new tokens, then 128 greedy tokens are generated.
tg = decode tok/s, pp@16K = 512-token prompt appended to the restored 16K slot, heartbeat = 300-token append
to the restored 16K slot + 64 generated tokens (TTFT / total seconds). Median of 3, 60 s cooldown between
runs. VRAM peak = llama-server process dedicated usage (MB), plus shared (system) memory in brackets.
Step 0 llama-bench numbers are in FACTS.md.

| time | commit/label | change | tg@0 | tg@16K | tg@32K | heartbeat | pp@16K | VRAM peak | correctness | verdict | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 15:24 | 5500m-start (srv) | baseline, start build | 8.86 | 7.98 | n/a (32K not resident; llama-bench OOM) | 16.53 s / 24.87 s | 16.37 | 6081 [+581] | KLD 0.002896 (ref f16 KV) | baseline | same protocol as all experiments |
| 16:05 | 37f35dd53 exp1 | avoid host-visible vidmem (256 MiB BAR heap) for device buffers | 14.40 | 11.64 | 6.5 @ -c 32768 (spills) | 16.87 s / 23.58 s | 16.95 | 6216 [+381] | KLD 0.002896 (=), 8/8, slot identical, tbo GET_ROWS/CPY/SSM_CONV/FA/MUL_MAT pass (GDN raw_gates fails pre-existing) | WIN +63% tg@0, +46% tg@16K | state GET_ROWS/CPY 243/251 us -> 29/25 us |
| 17:05 | 6771079d3 exp2 | GDN reads recurrent state row in place (rows mode on Vulkan, n_rs_seq=0 single-seq path) | 14.91 | 12.59 (5 reps) | - | 15.91 s / 21.46 s | 18.00 | 6231 [+431] | KLD 0.002896 (=), 8/8, slot identical (40-token resume), tbo GDN 45/45 | WIN +3.5% tg@0, +8% tg@16K, -9% hb total | GET_ROWS 97->49/token, GPU 71.5->67.3 ms |
| 17:30 | a751c6df8 exp3 | fuse GDN state write-back CPY into GDN kernel (GDN_STATE_CPY) | 15.47 | 12.90 | - | 15.84 s / 20.81 s | 19.28 | 6231 [+431] | KLD 0.002896 (=), 8/8, slot identical, tbo GDN 45/45 + new fusion test 5/5 | WIN +2.9% tg@0, +3.9% tg@16K (same-session A/B vs exp2: 15.04 / 12.42) | CPY 96->48/token, GPU 67.3->66.1 ms |
| 17:55 | exp4 (reverted) | PTQ1_0 integer-dot GEMM (MMQ) for prefill; dword trit decode in repack | 15.46 | 12.93 | - | 12.13 s / 17.12 s | 25.82 | 6236 [+431] | **KLD 0.004325 vs 0.002896 (FAIL, +16 sigma)**, 8/8, tbo MUL_MAT ptq1_0 71/71 | LOSS (correctness): pp +34%, hb TTFT -23% but q8_1 activations in prefill raise KLD | GEMM 1.99 -> 3.50 TFLOPS; patch kept in patches/exp4-mmq-ptq1_0-final.patch |
| 18:25 | 13b1079f4 exp5 | PTQ1_0 dword decode in float mul_mm loader (prefill, full precision) | 15.24 | (unchanged) | - | 14.54 s / 20.38 s | 19.99 | 6231 [+431] | KLD 0.002896 (bit-identical), 8/8, slot identical, tbo MUL_MAT ptq1_0 71/71 | WIN pp +3.7%, hb TTFT -8% | GEMM 1.99 -> 2.36 TFLOPS |
| 18:40 | exp6 (reverted) | PTQ1_0 mat-vec geometry sweep: rows/WG {2,4,8,16}, wave32 vs wave64, subgroup vs 4x-subgroup WG | - | - | - | - | - | - | n/a (no code kept) | NO WIN: default 4 rows / wave64 / subgroup WG fastest on all shapes | isolated us (17408x5120 / 5120x17408 / 10240x5120): default 181/163/109; 2r 203/178/132; 8r 191/177/115; wave32 4r 206/195/123; large WG 276-346 |
| 19:25 | dda7aad3a exp7 | shared-memory FWHT for n=1024 on AMD (was one subgroup per row) | 16.07 | 13.24 | - | 14.58 s / 20.37 s | 19.67 | 6231 [+431] | KLD 0.002896 (=), 8/8, slot identical, tbo MUL_MAT_HADAMARD 27/27 | WIN +5.3% tg@0 (A/B), tg@16K +1.7..10% | FWHT 21.3 -> 7.7 us x306/token; GPU 66.1 -> 62.2 ms |
| 20:15 | 652b83fda exp8 | FA tuning for hsk=256 on AMD RDNA: block_rows 16->8 (prefill), d_split 8->16 (decode) | 16.63 | 13.91 | - | 9.76 s / 14.36 s | 32.32 | 6197 [+431] | KLD 0.002896 (=, ub 256 rerun after desktop-VRAM OOM), 8/8, slot identical, tbo FA 5150/5150 | WIN tg@16K +5%, pp@16K +64%, hb TTFT -33% | FA prefill 16K 666->224 ms/layer (was register-spilling), decode 863->768 us |
| 20:35 | exp9 (reverted, near-miss) | GDN raw gates on Vulkan (sigmoid/softplus in GDN kernel; -192 dispatches/token) | 16.60 | 13.97 | - | - | - | ~6200 | tbo GDN 49/49 (raw-gates cases now pass) | NO WIN: A/B vs exp8 16.64 / 13.91 (-0.2% / +0.4%) | GPU 59.2 ms/token (-~1 ms) but not visible end to end; patch kept in patches/exp9-gdn-raw-gates.patch for combining |
| 21:45 | bb9f8fda0 exp10 | 64x128 PTQ1_0 mat-mat tile on RDNA1, chosen by column padding (n=508 wide, n=296 medium) | (unchanged) | (unchanged) | - | 9.76 s / 14.36 s (=) | 34.58 | ~6200 | KLD 0.002896 (=), 8/8, slot identical, tbo MUL_MAT ptq1_0 71/71 | WIN pp@16K +6.6% (A/B 32.43) | GEMM 2.37 -> 2.63 TFLOPS; first try (wide for all n) hurt hb TTFT +2.3%; n>=512 rule never fired (server splits 508+4) |
| 21:55 | exp11 (reverted) | no f16 K/V dequant scratch for prefill FA on RDNA1 (VRAM track) | - | - | - | 9.64 s / 14.35 s | 35.11 | 6197 [+431] (same as exp10) | tbo FA 5150/5150 | NO WIN: A/B vs exp10 pp 34.90, hb 9.67/14.37; process VRAM identical after pp512@16K | synthetic FA perf used the scratch (+3.6% faster with it) but the model K/V views are not dense, so the real model never allocates it; nothing to save |
| 22:15 | exp12 (reverted) | FWHT (shared-memory variant) also writes q8_1_x4 for its mat-vec consumer + GDN raw gates (near-miss combo) | 16.61 | 13.85 | - | - | - | ~6200 | tbo new FWHT_PTQ1_MV 12/12, HADAMARD 27/27, GDN 49/49, MUL_MAT ptq1_0 71/71 | NO WIN: A/B vs exp10 16.61 / 13.75 (0% / +0.7%) | GPU 59.6 -> 59.0 ms: FWHT 7.7 -> 9.5 us (q8 write + barrier) and the graph optimizer runs several FWHTs before their consumers, so the single prealloc_y copy is overwritten; data-pointer dedupe also has a stale-reuse hazard. Skip-all-quantize probe bound: +5%. Patch kept. |
| 22:35 | exp13 (probe, no code) | GGML_VK_DISABLE_GRAPH_OPTIMIZE=1 (no node reordering) on exp10 build | 16.47 | 13.70 | - | - | - | - | n/a | NO WIN: same-session default 16.60 / 13.75 | reordering helps slightly; keeps default |
| 22:55 | exp14 (reverted) | FA mask optimization for single-token decode (skip mask loads + 3 barriers per 32-position block when all-zero) | 16.63 | 13.81 | - | - | - | - | tbo FA 5150/5150 | NO WIN: same binary, switch off 16.53 / 13.74 (+0.6% / +0.5%) | synthetic FA perf (random mask) 763 -> 772 us; real model within noise |
| 22:58 | exp15 (reverted) | medium mat-mat tile sweep at heartbeat batch size (n=296): 8 configs | - | - | - | - | - | - | tbo MUL_MAT ptq1_0 8/8 per config | NO WIN: default 64x64 tile fastest (2.22 / 2.30 TFLOPS for 5120x296x17408 / 17408x296x5120; alternatives 1.73-2.20) | 6th consecutive experiment without a win -> stop |
| 00:40 | 5500m-start (srv, re-run) | final comparison, same session: start build | 9.16 | 8.24 | n/a | 15.83 s / 23.57 s | 19.55 | 6081 [+581] | - | reference | back-to-back with the final build |
| 00:15 | bb9f8fda0 FINAL (srv) | final build (7 commits on 5500m-tuning) | 16.74 | 13.90 | n/a (32K not resident on this desktop) | 9.67 s / 14.33 s | 34.90 | 6197 [+431] | KLD 0.002896 (= start), 8/8, slot identical | +83% / +69% / -39% / +79% vs start | llama-bench tg128@0 9.30 -> 16.89, pp512@0 34.6 -> 49.7 |
