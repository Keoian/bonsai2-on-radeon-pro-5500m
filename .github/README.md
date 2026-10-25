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

| Server decode, 3-4 prompts, 128-256 tokens | No MTP | MTP |
|---|---|---|
| greedy | 14.2-14.6 tok/s | 16.8-19.7 tok/s, 70-96% of drafts accepted |
| temperature 1.0 (model card) | 14.1-14.2 tok/s | 17.96 tok/s, 76% accepted (17.31 and 70% with greedy drafts) |

At a temperature above 0 the head samples its guess at the request's temperature and top-p, and the server
accepts it with the speculative-sampling rule (min(1, p/q)), which accepts more drafts than matching the
argmax. `LLAMA_SPEC_SAMPLE_DRAFT=0` turns that off.

`-Accept escape` (or `"speculative.accept": "escape"` per request, or the chat UI's "Draft acceptance" setting) is
faster but **approximate**: it keeps a draft unless the model's own sampler could never pick it there (probability 0
after penalties, top-k, top-p and min-p), so the output is no longer distributed exactly as without MTP. Port of
dual-p100-qwen-3-8-27b 09a243185, where quality matched exact on that repo's code, math and thinking evals.
`esc+<x>` also vetoes drafts below x times the top token's probability. Same 6 prompts and seeds, rules alternated:

| Preset | exact | escape |
|---|---|---|
| thinking (1.0 / 0.95 / top-k 20 / min-p 0.05) | 18.92 tok/s, 73.6% accepted | 19.88 tok/s, 89.3% accepted |
| thinking off (0.7 / 0.8 / top-k 20 / presence 1.5) | 16.90 tok/s, 73.5% | 17.47 tok/s, 86.1% |

The gain is smaller than on the P100 (+14-31%) because a step here drafts one token, so it can win at most 2.

The head, its KV cache and the delta-net rollback state take about 1 GB of VRAM, so the context drops from 64k
to 48k (52k is the most that fits). A Q8_0 head draws the same tokens and fits 40k. Drafting two tokens per
step is slower on this card (14.9 vs 18.0 tok/s): the ternary mat-vec decodes the weights once per 2 columns,
so verifying 3 tokens costs 128 ms against 86 ms for 2.

`-Lan` exposes an unauthenticated API. Only use it on a network you trust, or add `-- --api-key <key>`.

## Chat UI

[`bonsai2-5500m/webui/index.html`](../bonsai2-5500m/webui/index.html) is one file with no dependencies.
The launcher serves it in place of llama-server's built-in UI.

- Streams replies and shows the model's thinking in a collapsible block.
- Shows decode and prompt speed for every reply, taken from the server's own timings.
- Per-chat reasoning effort (Off / 512 / 2048 / 8192 / unlimited thinking tokens), sampling presets
  from the model card, system prompt, image attach, light/dark theme.
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
