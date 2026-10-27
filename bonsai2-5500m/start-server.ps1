<#
.SYNOPSIS
  Start llama-server for Bonsai 2 27B with the tuned Vulkan settings and serve the chat UI.

.EXAMPLE
  .\start-server.ps1                 # Vulkan, PTQ1_0, 64k context, KV cache q8_0 keys + q4_0 values
  .\start-server.ps1 -CacheK f16 -CacheV f16 -Ctx 16384   # full-precision KV cache; 16k is its limit in 8 GB
  .\start-server.ps1 -Lan            # also reachable from other machines on the LAN
  .\start-server.ps1 -Model pq2      # use the PQ2_0 file instead
  .\start-server.ps1 -Cpu            # CPU only (PQ2_0), for machines without a usable GPU
  .\start-server.ps1 -Parallel 2    # more concurrent chats (each slot costs VRAM; 1 fits the 8 GB 5500M)
  .\start-server.ps1 -Ctx 16384 -NoVision -- --reasoning-budget 2048   # extra llama-server flags after --
  .\start-server.ps1 -Mtp            # MTP speculative decoding: +20-35% decode, 48k context (run graft-mtp.py once first)
  .\start-server.ps1 -Mtp -Accept escape   # also keep drafts the model could have sampled (approximate, faster)

  .\start-server.ps1 -ModelDir D:\models   # where the .gguf files live (see below)

  Then open http://localhost:8080 in a browser.

  To keep a chat across a server restart: .\chat-cache.ps1 save before stopping, .\chat-cache.ps1 restore after
  starting again (same model and settings). The files go to -SlotDir, by default .\slots next to this script.

  Model files are looked up in: -ModelDir, then $env:BONSAI_MODEL_DIR, then .\models next to this
  script, then a "models" folder beside the repo checkout. The server binary is expected at
  <repo>\build\bin\llama-server.exe (the default CMake build folder).
#>
[CmdletBinding(PositionalBinding = $false)]
param(
    [ValidateSet("ptq1", "pq2")] [string] $Model = "ptq1",
    [switch] $Lan,
    [switch] $Cpu,
    [switch] $NoVision,
    [switch] $MmprojGpu,
    # MTP head from ProCreations/Ternary-Bonsai-2-27B-MTP (graft-mtp.py builds the file). The head, its KV cache
    # and the GDN rollback state cost ~1 GB of VRAM, so the default context drops to 48k (40k with a Q8_0 head).
    [switch] $Mtp,
    # With -Mtp: the default draft accept rule. exact keeps the model's output distribution; escape (approximate) also
    # keeps drafts the model could have sampled but did not. Requests and the chat UI can override it.
    [string] $Accept,
    [int]    $Ctx = 65536,
    # KV cache precision. On the 8 GB 5500M at 32k: q8_0/q8_0 fits but decode drops to ~9 tok/s (VRAM
    # nearly full); q8_0 keys + q4_0 values keeps ~14 tok/s. Keys are the precision-sensitive half.
    [ValidateSet("f16", "q8_0", "q4_0")] [string] $CacheK = "q8_0",
    [ValidateSet("f16", "q8_0", "q4_0")] [string] $CacheV = "q4_0",
    [int]    $Port = 8080,
    [int]    $Ngl = 99,
    [int]    $Parallel = 1,
    [string] $ModelDir,
    # where chat-cache.ps1 saves and restores chats (the server's slot save path)
    [string] $SlotDir = (Join-Path $PSScriptRoot "slots"),
    [Parameter(ValueFromRemainingArguments = $true)] [string[]] $Extra
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path $PSScriptRoot -Parent
$Bin      = Join-Path $RepoRoot "build\bin\llama-server.exe"
$WebUI    = Join-Path $PSScriptRoot "webui"

if (-not (Test-Path $Bin)) { throw "llama-server.exe not found at $Bin (build first, see .github/README.md)" }

$Candidates = @($ModelDir, $env:BONSAI_MODEL_DIR, (Join-Path $PSScriptRoot "models"), (Join-Path (Split-Path $RepoRoot -Parent) "models")) |
    Where-Object { $_ }
$Models = $Candidates | Where-Object {
        (Test-Path (Join-Path $_ "Ternary-Bonsai-2-27B-PTQ1_0.gguf")) -or (Test-Path (Join-Path $_ "Ternary-Bonsai-2-27B-PQ2_0.gguf"))
    } | Select-Object -First 1
if (-not $Models) { throw "no Bonsai 2 .gguf found. Looked in: $($Candidates -join '; '). Pass -ModelDir." }

if ($Cpu) { $Model = "pq2"; $Ngl = 0 }
$Gguf = if ($Model -eq "pq2") { "Ternary-Bonsai-2-27B-PQ2_0.gguf" } else { "Ternary-Bonsai-2-27B-PTQ1_0.gguf" }
if ($Mtp) {
    if ($Model -ne "ptq1") { throw "-Mtp needs the PTQ1_0 model on the GPU" }
    $MtpGguf = "Ternary-Bonsai-2-27B-PTQ1_0-MTP-Q4_K.gguf", "Ternary-Bonsai-2-27B-PTQ1_0-MTP-Q8_0.gguf" |
        Where-Object { Test-Path (Join-Path $Models $_) } | Select-Object -First 1
    if (-not $MtpGguf) { throw "no MTP model in $Models. Build it once with: python $(Join-Path $PSScriptRoot 'graft-mtp.py')" }
    $Gguf = $MtpGguf
    if (-not $PSBoundParameters.ContainsKey("Ctx")) { $Ctx = if ($Gguf -like "*Q4_K*") { 49152 } else { 40960 } }
}
$GgufPath = Join-Path $Models $Gguf
if (-not (Test-Path $GgufPath)) { throw "model not found: $GgufPath" }

$BindHost = if ($Lan) { "0.0.0.0" } else { "127.0.0.1" }

# Sampling per the Bonsai 2 model card (thinking mode). The web UI overrides these per request.
$Args_ = @(
    "-m", $GgufPath,
    "--host", $BindHost, "--port", "$Port",
    "-ngl", "$Ngl", "-fa", "on",
    "-c", "$Ctx",
    "-np", "$Parallel",
    "-ctk", $CacheK, "-ctv", $CacheV,
    "--temp", "1.0", "--top-p", "0.95", "--top-k", "20",
    "--jinja",
    "--path", $WebUI,
    "--slots",
    "--slot-save-path", $SlotDir
)
New-Item -ItemType Directory -Force $SlotDir | Out-Null
if ($Cpu) { $Args_ += @("-t", [Math]::Max(1, [Environment]::ProcessorCount / 2)) }
if ($Mtp) {
    # One draft token per step: two are slower here (16.4 vs 26.7 tok/s, see .github/README.md). -ctkd/-ctvd give the draft KV
    # cache the target's types (it is f16 otherwise), and a 128-row draft batch cuts its compute buffer from 130 to 35 MB.
    $Args_ += @("--spec-type", "draft-mtp", "--spec-draft-n-max", "1", "-ctkd", $CacheK, "-ctvd", $CacheV)
    if (-not $env:LLAMA_ARG_SPEC_DRAFT_UBATCH) { $env:LLAMA_ARG_SPEC_DRAFT_UBATCH = "128" }
    if ($Accept) { $Args_ += @("--spec-accept", $Accept) }
}
if (-not $NoVision) {
    $Mmproj = Join-Path $Models "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf"
    if (Test-Path $Mmproj) {
        $Args_ += @("--mmproj", $Mmproj, "--image-max-tokens", "1024")
        # The 0.63 GB projector does not fit next to the 27B weights on an 8 GB card; keep it in RAM
        # (slower image encode, no effect on text). -MmprojGpu puts it back on the GPU.
        if (-not $MmprojGpu) { $Args_ += "--no-mmproj-offload" }
    }
}
if ($Extra) { $Args_ += ($Extra | Where-Object { $_ -ne "--" }) }

# Hybrid graphics (GMUX mode 5): the Intel iGPU is also a Vulkan device and is listed first, and
# llama.cpp would split the model across both. Pin the Radeon by name unless -dev/--device was given.
if (-not $Cpu -and -not ($Args_ | Where-Object { $_ -in "-dev", "--device" })) {
    $ErrorActionPreference = "Continue"
    $Radeon = & (Join-Path $RepoRoot "build\bin\llama-bench.exe") --list-devices 2>&1 |
        ForEach-Object { "$_" } | Select-String '^\s*(Vulkan\d+): .*Radeon' | Select-Object -First 1
    $ErrorActionPreference = "Stop"
    if ($Radeon) { $Args_ += @("-dev", $Radeon.Matches[0].Groups[1].Value) }
}

Write-Host ""
Write-Host "=== Bonsai 2 27B / llama-server ===" -ForegroundColor Green
Write-Host "  Model:   $Gguf"
Write-Host ("  Backend: " + $(if ($Ngl -gt 0) { "Vulkan, -ngl $Ngl" + $(if ($Radeon) { ", device $($Radeon.Matches[0].Groups[1].Value)" } else { "" }) } else { "CPU" }))
Write-Host "  Context: $Ctx (KV cache K=$CacheK V=$CacheV)"
if ($Mtp) { Write-Host ("  Spec:    MTP, 1 draft token per step, accept rule " + $(if ($Accept) { $Accept } else { "exact" })) }
Write-Host ("  UI:      http://localhost:$Port" + $(if ($Lan) { "  (LAN: http://$((Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.*' } | Select-Object -First 1).IPAddress):$Port)" } else { "" }))
Write-Host "  API:     http://localhost:$Port/v1/chat/completions"
Write-Host "  Chats:   .\chat-cache.ps1 save / restore (files in $SlotDir)"
Write-Host "  Ctrl+C stops the server."
Write-Host ""

# llama-server logs to stderr; "Stop" would abort on the first log line when output is redirected.
$ErrorActionPreference = "Continue"
& $Bin @Args_
exit $LASTEXITCODE
