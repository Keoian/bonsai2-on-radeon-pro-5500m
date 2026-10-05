<#
.SYNOPSIS
  Bonsai 2 27B with a 32k context. All other arguments pass through to start-server.ps1.

.DESCRIPTION
  32k (q8_0 keys + q4_0 values) needs about 6.9 GB of VRAM for llama.cpp, and Windows lets applications use
  about 7.8 GB of the 5500M's 8 GB. Because the 5500M also drives the display, 32k only stays resident when the
  desktop and other apps hold under ~0.9 GB (close Chrome, Remote Desktop, etc., or turn off their hardware
  acceleration). Otherwise the KV cache allocation fails or part of the model spills to system RAM and decoding
  slows sharply. This script checks current VRAM use first and warns; it still starts the server.
#>
$LimitMB  = 7800   # usable dedicated VRAM on the 5500M under Windows (measured)
$NeedMB   = 6900   # llama.cpp at 32k with q8_0 / q4_0 KV
try {
    $inst = (Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage' -ErrorAction Stop).CounterSamples |
        Sort-Object CookedValue -Descending | Select-Object -First 1
    $usedMB = [int]($inst.CookedValue / 1MB)
    $freeMB = $LimitMB - $usedMB
    if ($freeMB -lt $NeedMB) {
        Write-Host ""
        Write-Host ("WARNING: other apps hold {0} MB of VRAM; 32k needs ~{1} MB and only ~{2} MB is available." -f $usedMB, $NeedMB, $freeMB) -ForegroundColor Yellow
        Write-Host "Expect an out-of-memory error or slow, spilled decoding. Largest GPU memory users:" -ForegroundColor Yellow
        (Get-Counter '\GPU Process Memory(*)\Dedicated Usage').CounterSamples |
            Where-Object CookedValue -gt 50MB | Sort-Object CookedValue -Descending | Select-Object -First 6 |
            ForEach-Object {
                $procId = [int]($_.InstanceName -replace '^pid_(\d+)_.*', '$1')
                $name = (Get-Process -Id $procId -ErrorAction SilentlyContinue).ProcessName
                Write-Host ("  {0,6} MB  {1}" -f [int]($_.CookedValue / 1MB), $name)
            }
        Write-Host "(dwm, the desktop compositor, always stays; it also counts other apps' windows.) Use start-server-16k.ps1 otherwise." -ForegroundColor Yellow
        Write-Host ""
    } else {
        Write-Host ("VRAM check: other apps hold {0} MB, ~{1} MB available for the 32k context." -f $usedMB, $freeMB)
    }
} catch {
    Write-Host "VRAM check skipped (GPU performance counters unavailable)."
}

& (Join-Path $PSScriptRoot "start-server.ps1") -Ctx 32768 @args
exit $LASTEXITCODE
