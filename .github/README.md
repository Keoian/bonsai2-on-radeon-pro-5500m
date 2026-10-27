# Bonsai 2 27B on a Radeon Pro 5500M (Vulkan, Windows)

A build of the [PrismML llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp) that runs
[Ternary Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) at about
**14.5 tokens/s** on an 8 GB AMD Radeon Pro 5500M, up from about 1.7 tokens/s on the stock fork.
It adds a one-command launcher and a small single-file chat UI.

Nothing here is new kernel work. It is upstream `prism` with two open Vulkan pull requests merged in:

| PR | Author | What it does |
|---|---|---|
| [#188](https://github.com/PrismML-Eng/llama.cpp/pull/188) | @alhnesn | Integer-dot mat-vec kernel for `PTQ1_0`, and Vulkan support for `PQ2_0`. This is where the speedup comes from. |
| [#187](https://github.com/PrismML-Eng/llama.cpp/pull/187) | @MrFadiAi | Trit lookup table, vectorized dequant, MUL+FWHT fusion. Roughly +12% prompt processing. |

The slow path they replace is tracked upstream in issues
[#185](https://github.com/PrismML-Eng/llama.cpp/issues/185),
[#186](https://github.com/PrismML-Eng/llama.cpp/issues/186) and
[#201](https://github.com/PrismML-Eng/llama.cpp/issues/201).
Once both PRs merge, plain upstream `prism` will be just as fast and this repo is only the launcher and UI.

## Measured

MacBook Pro 16" (2019): i9-9980HK, Radeon Pro 5500M 8 GB, 32 GB RAM, Windows 11, AMD Boot Camp
driver 32.0.12019.1028. All layers on the GPU.

| Build | File | llama-bench tg32 | llama-bench pp128 | Real chat decode (llama-server timings) |
|---|---|---|---|---|
| upstream `prism` (b10687) | PTQ1_0 | 1.56 tok/s | 32 t/s | ~1.7 tok/s |
| this repo | PTQ1_0 | 9.3 tok/s | 35 t/s | **14.2 to 14.7 tok/s** |
| this repo | PQ2_0 | 8.9 tok/s | 48 t/s | does not fit in 8 GB when fully offloaded |

`llama-bench`'s short decode test reads lower than real generation on this card. The number to trust is
`timings.predicted_per_second` from the server. `GGML_VK_FORCE_MMVQ=1` and `-fa off` were within 2% of the
defaults. The driver must report `int dot: 1` in the startup log, since PR #188's kernel needs it.

## Build (Windows, MSVC + Vulkan)

Needs Visual Studio 2022 Build Tools (C++), CMake, Ninja, and the Vulkan SDK. Run from a
"x64 Native Tools" / Developer PowerShell prompt, so the MSVC environment is loaded:

```
git clone https://github.com/Keoian/bonsai2-on-radeon-pro-5500m.git
cd bonsai2-on-radeon-pro-5500m
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build build -j 16
```

Get the model from [prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf):
`Ternary-Bonsai-2-27B-PTQ1_0.gguf` (5.9 GB) for 8 GB cards, and optionally
`Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf` for image input. Put them in `bonsai2-5500m\models\`
(ignored by git) or anywhere you like and pass `-ModelDir`.

Stock llama.cpp cannot run these files. Bonsai 2 needs the fork's Hadamard activation transform.

## Run

```
.\bonsai2-5500m\start-server.ps1               # PTQ1_0 on Vulkan, 64k context
.\bonsai2-5500m\start-server.ps1 -Lan          # listen on 0.0.0.0 for other machines on the LAN
.\bonsai2-5500m\start-server.ps1 -Cpu          # CPU-only fallback (uses PQ2_0)
.\bonsai2-5500m\start-server.ps1 --reasoning-budget 2048    # unknown flags pass straight to llama-server
```

Then open http://localhost:8080. The launcher's defaults exist to fit in 8 GB of VRAM:

- `-np 1`: every server slot allocates its own recurrent-state cache, and the default of 4 runs out of memory.
- `--no-mmproj-offload`: the 0.63 GB vision projector stays in system RAM. Text speed is unaffected and
  image encoding is slower. `-MmprojGpu` puts it back on the GPU if you have room.
- 64k context with a mixed KV cache: 8-bit keys and 4-bit values (`-CacheK q8_0 -CacheV q4_0`).
  Keys are the half that loses accuracy when compressed, so they keep 8 bits.
- Keep Resizable BAR off. A 4 GB BAR makes the AMD driver split VRAM into two ~4 GB heaps; ggml-vulkan fills the
  host-visible one past its budget and Windows moves about 1 GB of it to system RAM (decode drops to ~4.4 tok/s).
  With the default 256 MB BAR, commit 24eebb5e7 keeps every buffer in the 7.7 GB device-local heap.

KV cache options measured on this card (same short prompt, fresh server each time):

| Keys / values | Context | Fits? | Decode |
|---|---|---|---|
| f16 / f16 | 16k | yes | 14.1 tok/s |
| f16 / f16 | 20k, 24k | out of memory | |
| q8_0 / q8_0 | 16k | yes | 14.1 tok/s |
| q8_0 / q8_0 | 32k | yes, but VRAM is nearly full | 9.1 tok/s |
| q8_0 / q4_0 | 32k | yes | 14.2 tok/s |
| **q8_0 / q4_0 (default)** | **64k** | **yes, 7.45 GB in VRAM** | **14.5 tok/s** |
| q8_0 / q4_0 | 68k | yes, 7.56 GB in VRAM (little headroom) | 14.7 tok/s |
| q8_0 / q4_0 | 72k, 76k, 82k | out of memory | |
| q4_0 / q4_0 | 32k | yes | 14.2 tok/s |

Quality cost of the default cache, measured as WikiText-2 perplexity (lower is better, same text for both):

| Window | f16 / f16 | q8_0 / q4_0 | Change |
|---|---|---|---|
| 2k tokens, 10 chunks | 7.1790 | 7.1737 | -0.07% (noise) |
| 16k tokens, 1 chunk | 4.6921 | 4.7092 | +0.36% |

The 16k window is the most full precision can hold on this card, so it is the fairest long-context comparison.
A 21,500-token recall test also passed with the default: the model found a code word planted near the start.

For full-precision cache use `-CacheK f16 -CacheV f16 -Ctx 16384`. On a card with more VRAM, raise `-Ctx`.

### Keeping a chat across a server restart

The server keeps recent chats in system RAM (up to 8 GB, one context's worth of tokens), but loses them when it
stops. Without them, the first message after a restart reads the whole chat again, at about 25-35 tokens/s here
(15+ minutes for a 30k-token chat). The launcher passes `--slot-save-path` (`-SlotDir`, default `bonsai2-5500m\slots`,
ignored by git), so the current chat can be saved to disk and loaded back:

```
.\bonsai2-5500m\chat-cache.ps1 save          # before stopping the server: writes slots\chat.bin
.\bonsai2-5500m\chat-cache.ps1 restore       # after starting it again, before the next message
.\bonsai2-5500m\chat-cache.ps1 save work     # a named file, slots\work.bin
.\bonsai2-5500m\chat-cache.ps1 list
```

A file holds the KV cache, the Gated DeltaNet checkpoints (about 150 MiB each, so 1-2 GB for a long chat) and the MTP
draft state, and it only loads into a server started with the same model, KV cache types and `-Mtp` setting. On the
5500M a short chat saved (601 MiB) in 0.7 s and restored in 0.5 s. This uses two upstream llama.cpp server changes,
cherry-picked here: checkpoints in the slot file (#26004) and the draft state in the slot file (#30574).

### MTP speculative decoding

`-Mtp` adds a one-layer draft head (multi-token prediction) that guesses the next token. The full model checks
the guess in the same pass and keeps it only where its own sampler agrees, so the output keeps the model's
distribution. Greedy text can still change at near-ties (two candidates within ~0.01 probability), because a
2-token verify batch rounds slightly differently from a 1-token decode. Build the model file once. The script
downloads only the 430 MiB head of [ProCreations/Ternary-Bonsai-2-27B-MTP](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-MTP)
and copies the rest from your PTQ1_0 file, then requantizes the head to Q4_K (`--head-type q8_0` keeps it as is):

```
python bonsai2-5500m\graft-mtp.py
.\bonsai2-5500m\start-server.ps1 -Mtp          # 48k context
```

| Server decode at a23daef20, short context, cool GPU | No MTP | MTP |
|---|---|---|
| greedy, 4 prompts x 128 tokens, thinking off | 19.15 tok/s | 26.96 tok/s |
| temperature 1.0 / top-p 0.95 / top-k 20 / min-p 0.05, 4 prompts x 200 tokens x 3 rounds | | 26.55 tok/s, 75.3% accepted |
| same sampling, thinking on, 4 prompts x 384 tokens (fb1d2f0df) | | 26.7 tok/s, 74.3% accepted |

Without MTP the sampling settings barely matter: llama-bench tg64 gives 19.05-19.15 tok/s on the same build. Before
the RDNA1 decode kernels of 2026-10-26 (decode flash attention, the ternary mat-vec, the Hadamard/norm/gate fusions)
the same kind of runs gave 14.2-14.6 tok/s without MTP and 16.8-19.7 with it (greedy), and 17.96 tok/s with MTP at
temperature 1.0. On real chat turns with thinking (1.4k-9.8k tokens of context) 70.6-73.9% of drafts were accepted.

Long replies are slower, because the laptop throttles. After about 80 s of nonstop decoding the GPU falls into a
~20 s cycle with ~9 s at 6-8 tok/s (without MTP: 17.3 tok/s cool, about 9.5 tok/s on average in the cycling phase).
On a fan pad, 60 short replies in a row averaged 15.0 tok/s against 11.2 without it (the battery also stopped
charging in between). With MTP on the fan pad, 14 back-to-back 512-token replies averaged 20.3 tok/s (26.6 for the
first, about 21 after two minutes), and one 16-minute, 18,273-token reply at 10k-28k tokens of context averaged
18.6 tok/s. A longer context also slows attention.

At a temperature above 0 the head samples its guess at the request's temperature and top-p, and the server
accepts it with the speculative-sampling rule (min(1, p/q)), which accepts more drafts than matching the
argmax. `LLAMA_SPEC_SAMPLE_DRAFT=0` turns that off.

`-Accept escape` (or `"speculative.accept": "escape"` per request, or the chat UI's "Draft acceptance" setting) is
faster but **approximate**: it keeps a draft unless the model's own sampler could never pick it there (probability 0
after penalties, top-k, top-p and min-p), so the output is no longer distributed exactly as without MTP. Port of
dual-p100-qwen-3-8-27b 09a243185, where quality matched exact on that repo's code, math and thinking evals.
`esc+<x>` also vetoes drafts below x times the top token's probability. Same 6 prompts and seeds, rules alternated
(measured at eeb36708b, before the 2026-10-26 decode kernels, so the absolute numbers are lower than today's):

| Preset | exact | escape |
|---|---|---|
| thinking (1.0 / 0.95 / top-k 20 / min-p 0.05) | 18.92 tok/s, 73.6% accepted | 19.88 tok/s, 89.3% accepted |
| thinking off (0.7 / 0.8 / top-k 20 / presence 1.5) | 16.90 tok/s, 73.5% | 17.47 tok/s, 86.1% |

The gain is smaller than on the P100 (+14-31%) because a step here drafts one token, so it can win at most 2.

The head, its KV cache and the delta-net rollback state take about 1 GB of VRAM, so the context drops from 64k
to 48k (52k is the most that fits). A Q8_0 head draws the same tokens and fits 40k.

#### Why one draft token per step

Two draft tokens per step are slower on this card, although the head guesses the second token well. Measured at
fb1d2f0df with 4 seeded prompts x 384 tokens, thinking on, temperature 1.0 / top-p 0.95 / top-k 20 / min-p 0.05,
1 and 2 drafts alternated over two rounds (both rounds gave the same counts):

| Drafts per step | 1 | 2 |
|---|---|---|
| Decode | **26.7 tok/s** | 16.4 tok/s |
| Drafts accepted | 74.3% | first 77.0%, second 65.6% (when the first was kept) |
| Tokens per step | 1.74 | 2.27 |
| Time per step | 65 ms | 138 ms |

A step costs one pass of the full model over the last token plus the drafts, and the pass gets much dearer at
3 tokens (llama-bench, one batch):

| Tokens in the pass | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| Time | 51.7 ms | 57.4 ms | 116.6 ms | 134.2 ms |

- The ternary mat-vec kernel for RDNA1 (`mul_mat_vecq_ptq1_0_w32.comp`) takes 1 or 2 columns: it decodes each
  weight block once and uses it for both. A 3-token verify falls back to the general kernel and costs twice as much.
- The head runs once per draft token, each time with a sampling round trip on the CPU: about 22 ms for two drafts,
  8 ms for one. Chained drafts (`LLAMA_MTP_CHAIN=1`, all steps in one decode) only work for greedy drafts, not for
  the sampled drafts the exact rule uses at a temperature above 0.

To win, a two-draft step has to take less than 65 x 2.27 / 1.74 = 85 ms. A 3-column version of the kernel might
verify in about 63 ms (a guess from the 1 -> 2 column cost), which with 22 ms of drafting is still about 85 ms:
no gain unless the drafting also gets much cheaper.

`-Lan` exposes an unauthenticated API. Only use it on a network you trust, or add `-- --api-key <key>`.

## Chat UI

[`bonsai2-5500m/webui/index.html`](../bonsai2-5500m/webui/index.html) is one file with no dependencies.
The launcher serves it in place of llama-server's built-in UI.

- Streams replies and shows the model's thinking in a collapsible block.
- Shows decode and prompt speed for every reply, taken from the server's own timings.
- Per-chat reasoning effort (Off / 512 / 2048 / 8192 / unlimited thinking tokens), sampling presets
  from the model card, system prompt, image attach, light/dark theme.
- "Draft acceptance" for an MTP server: server default, exact, or escape (see above).
- "Re-read each reply without its thinking": right after it (default) or with your next message (the old behavior).
  The UI sends earlier replies without their thinking, so the next prompt no longer matches what the server cached
  from the start of the last reply. The model's Gated DeltaNet layers cannot rewind, so without the re-read every
  turn first reads the whole last reply again (once 12 minutes for a reply that kept thinking in its visible text).
  With the default, the UI has the server read the reply as the next turn will send it (`max_tokens: 0`) while you
  read it, and your next message only needs your new text read. The stats line then adds "re-read for the next
  turn: N tok in S s". The cost is GPU work, and heat, for a while after each reply.
- Settings and the current chat are kept in the browser's local storage only.

If a browser shows an old page at `localhost:8080`, open `http://127.0.0.1:8080` or clear the site data.
llama-server's built-in UI can leave a service worker behind.

## Keeping up with upstream

```
git remote add upstream https://github.com/PrismML-Eng/llama.cpp.git
git fetch upstream
git merge upstream/prism
```

**Upstream CI is deliberately removed.** `.github/workflows/` is deleted in this repo. Left in place, those
workflows would run daily builds here and push Docker images and tags to this account's GitHub
Container Registry. When an upstream merge conflicts on a workflow file, drop it again:

```
git rm -r --quiet .github/workflows
git commit
```

## License

llama.cpp and the PrismML fork are MIT licensed, see [LICENSE](../LICENSE). The Bonsai 2 model weights
have their own license on their Hugging Face page.
