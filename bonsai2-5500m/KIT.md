# Bonsai 2 27B (ternary) — offline kit

Assembled 2026-09-17. Everything needed to run Bonsai 2 27B on Windows x64 machines without internet.

## What is here

| Path | What |
|---|---|
| `llama.cpp/` | Git clone of the PrismML fork. Checked-out branch `main` = upstream `prism` (9a9394a89, build 10709) + PR #188 (integer-dot Vulkan mat-vec for PTQ1_0, Vulkan PQ2_0 kernels) + PR #187 (PTQ1_0 trit table, MUL+FWHT fusion) + the launcher and UI in `bonsai2-5500m/`. Remote `origin` = https://github.com/Keoian/bonsai2-on-radeon-pro-5500m (public), `upstream` = PrismML (fetch only). Upstream CI workflows are deleted in this branch on purpose; see `.github/README.md`. Stock llama.cpp will NOT load these models. |
| `llama.cpp/build/bin/` | Built 2026-09-18 from the same code as `main` with MSVC 2022 + Ninja: CPU (all x86 variants, auto-selected at runtime) + Vulkan backend. This is the deployable folder. |
| `llama.cpp/build/bin-b10687-baseline/` | The previous (2026-09-17, unpatched) build, kept as a fallback. Vulkan decode was 6x slower with it. |
| `llama.cpp/bonsai2-5500m/` | Launchers (`start-server.ps1`, plus `start-server-16k.ps1` / `start-server-32k.ps1` presets) and single-file chat UI (`webui/index.html`). |
| `start-server.ps1`, `start-server-16k.ps1`, `start-server-32k.ps1` | Forwarders to the scripts of the same name in `llama.cpp/bonsai2-5500m/`, same arguments. |
| `models/Ternary-Bonsai-2-27B-PQ2_0.gguf` | 7.2 GB. Preferred on CPU, CUDA, HIP, Metal. Now also runs on Vulkan with this build (faster prefill, slightly slower decode than PTQ1_0) but does NOT fit fully offloaded in 8 GB VRAM. |
| `models/Ternary-Bonsai-2-27B-PTQ1_0.gguf` | 5.9 GB. Use this for Vulkan GPUs (AMD/Intel/NVIDIA via Vulkan) and for any 8 GB card. Also works on CPU/CUDA. |
| `models/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf` | 0.6 GB vision tower, only needed for image input. |
| `models/SHA256SUMS` | Verify with `sha256sum -c SHA256SUMS` (all three verified OK on download). |
| `deploy/vc_redist.x64.exe` | MSVC runtime installer, needed on target machines that lack it. |
| `Bonsai-demo/` | PrismML docs: TOOLS.md, VISION.md, SPECULATIVE.md, KV-CACHE.md, whitepapers. |

## Deploy to another Windows x64 machine

Copy `llama.cpp/build/bin/`, `models/`, and `deploy/vc_redist.x64.exe`. Run vc_redist if `llama-cli.exe` complains about missing VCRUNTIME/MSVCP DLLs. GPU machines need a Vulkan-capable driver (any recent AMD/NVIDIA/Intel driver). No Vulkan SDK is needed at runtime.

## Run (this laptop / any Vulkan machine)

```
.\start-server-16k.ps1        # PTQ1_0 on Vulkan, 16k context, UI at http://localhost:8080 (same as .\start-server.ps1)
.\start-server-32k.ps1        # 32k context: close Chrome / Remote Desktop first; warns if VRAM is short
.\start-server.ps1 -Lan       # bind 0.0.0.0 so other LAN machines can open the UI
.\start-server.ps1 -Cpu       # CPU-only fallback (PQ2_0)
.\start-server.ps1 --reasoning-budget 2048    # unknown flags pass straight to llama-server
```

What the launcher sets and why (8 GB VRAM is tight): `-np 1` (each server slot costs a recurrent-state
cache; 4 slots OOM), `--no-mmproj-offload` (the 0.63 GB vision projector stays in RAM; text speed is
unaffected, image encode is slower; `-MmprojGpu` reverses it), `-fa on`, `--jinja`, model-card sampling.
32k context fits with a mixed KV cache (q8_0 keys, q4_0 values) at full speed. A full-precision cache tops out at 16k, and q8_0 for both at 32k fits but slows decode to ~9 tok/s. Details in `llama.cpp/.github/README.md`.

Browser note: if a browser shows an old page at localhost:8080, open http://127.0.0.1:8080 instead or
clear site data; llama-server's built-in UI may have left a service worker behind.

## Run (manual commands)

CPU only (PQ2_0):
```
llama-cli.exe -m models\Ternary-Bonsai-2-27B-PQ2_0.gguf -ngl 0 -c 32768 --temp 1.0 --top-p 0.95 --top-k 20
```

Vulkan GPU, full offload (PTQ1_0, needs ~6 GB VRAM + context):
```
llama-cli.exe -m models\Ternary-Bonsai-2-27B-PTQ1_0.gguf -ngl 99 -fa on -c 32768 -ctk q8_0 -ctv q4_0 --temp 1.0 --top-p 0.95 --top-k 20
```

Partial offload for smaller GPUs: set `-ngl` to a layer count instead of 99.

OpenAI-compatible server for the LAN:
```
llama-server.exe -m models\Ternary-Bonsai-2-27B-PTQ1_0.gguf -ngl 99 -fa on -c 32768 -ctk q8_0 -ctv q4_0 -np 1 --host 0.0.0.0 --port 8080
```

Vision (add the mmproj):
```
llama-mtmd-cli.exe -m models\Ternary-Bonsai-2-27B-PQ2_0.gguf --mmproj models\Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf --image pic.jpg -p "Describe this."
```

Sampling per the model card: thinking mode temp 1.0 / top-p 0.95 / top-k 20; non-thinking temp 0.7 / top-p 0.8 / top-k 20 / presence 1.5. The model thinks by default at `xhigh` effort; see Bonsai-demo/TOOLS.md for reasoning budgets and tool calling.

## Rebuild from source (offline, on a machine with VS 2022 Build Tools, CMake, Ninja, Vulkan SDK)

```
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cd llama.cpp
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build build -j 16
```
Drop `-DGGML_VULKAN=ON` for a CPU-only build (no Vulkan SDK needed). CUDA builds need the CUDA toolkit (not installed here; add `-DGGML_CUDA=ON`).

## Measured on this laptop (i9-9980HK, Radeon Pro 5500M 8 GB, 32 GB RAM)

Driver 32.0.12019.1028 (AMD Boot Camp, Oct 2024). Vulkan reports `int dot: 1`, which PR #188 needs.

| Build | Backend | File | llama-bench tg32 | llama-bench pp128 | llama-server decode (chat) |
|---|---|---|---|---|---|
| b10687 unpatched | Vulkan, -ngl 99 | PTQ1_0 | 1.56 tok/s | 32 t/s | ~1.7 tok/s |
| + PR #188 | Vulkan, -ngl 99 | PTQ1_0 | 9.25 tok/s | 32 t/s | 14.5 tok/s |
| + PR #188 + #187 (current) | Vulkan, -ngl 99 | PTQ1_0 | 9.3 tok/s | 35 t/s | **14.2-14.7 tok/s** |
| + PR #188 + #187 | Vulkan, -ngl 99 | PQ2_0 | 8.9 tok/s | 48 t/s | OOM at -c 4096 (does not fit 8 GB) |
| b10687 | CPU, 8 threads | PQ2_0 | ~0.5 tok/s | | |

llama-bench's tg32 reads lower than real generation here (it includes per-run setup); the server
`timings.predicted_per_second` on real chats is the number to quote. `GGML_VK_FORCE_MMVQ=1` and `-fa off`
were within 2% of default, so the launcher leaves them alone.

Upstream status (2026-09-18): both PRs were open on github.com/PrismML-Eng/llama.cpp (#188 by alhnesn,
#187 by MrFadiAi; issues #185/#186/#201 track the slow Vulkan decode). When they merge, `git fetch` +
rebuild from `prism` replaces this local branch.
