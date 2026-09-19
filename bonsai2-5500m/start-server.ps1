<#
.SYNOPSIS
  Start llama-server for Bonsai 2 27B with the tuned Vulkan settings and serve the chat UI.

.EXAMPLE
  .\start-server.ps1                 # Vulkan, PTQ1_0, localhost:8080
  .\start-server.ps1 -Lan            # also reachable from other machines on the LAN
  .\start-server.ps1 -Model pq2      # use the PQ2_0 file instead
  .\start-server.ps1 -Cpu            # CPU only (PQ2_0), for machines without a usable GPU
  .\start-server.ps1 -Parallel 2    # more concurrent chats (each slot costs VRAM; 1 fits the 8 GB 5500M)
  .\start-server.ps1 -Ctx 16384 -NoVision -- --reasoning-budget 2048   # extra llama-server flags after --

  .\start-server.ps1 -ModelDir D:\models   # where the .gguf files live (see below)

  Then open http://localhost:8080 in a browser.

  Model files are looked up in: -ModelDir, then $env:BONSAI_MODEL_DIR, then .\models next to this
  script, then a "models" folder beside the repo checkout. The server binary is expected at
  <repo>\build\bin\llama-server.exe (the default CMake build folder).
#>
[CmdletBinding()]
param(
    [ValidateSet("ptq1", "pq2")] [string] $Model = "ptq1",
    [switch] $Lan,
    [switch] $Cpu,
    [switch] $NoVision,
    [switch] $MmprojGpu,
    [int]    $Ctx = 16384,
    [int]    $Port = 8080,
    [int]    $Ngl = 99,
    [int]    $Parallel = 1,
    [string] $ModelDir,
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
    "--temp", "1.0", "--top-p", "0.95", "--top-k", "20",
    "--jinja",
    "--path", $WebUI,
    "--slots"
)
if ($Cpu) { $Args_ += @("-t", [Math]::Max(1, [Environment]::ProcessorCount / 2)) }
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

Write-Host ""
Write-Host "=== Bonsai 2 27B / llama-server ===" -ForegroundColor Green
Write-Host "  Model:   $Gguf"
Write-Host ("  Backend: " + $(if ($Ngl -gt 0) { "Vulkan, -ngl $Ngl" } else { "CPU" }))
Write-Host "  Context: $Ctx"
Write-Host ("  UI:      http://localhost:$Port" + $(if ($Lan) { "  (LAN: http://$((Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.*' } | Select-Object -First 1).IPAddress):$Port)" } else { "" }))
Write-Host "  API:     http://localhost:$Port/v1/chat/completions"
Write-Host "  Ctrl+C stops the server."
Write-Host ""

# llama-server logs to stderr; "Stop" would abort on the first log line when output is redirected.
$ErrorActionPreference = "Continue"
& $Bin @Args_
exit $LASTEXITCODE
