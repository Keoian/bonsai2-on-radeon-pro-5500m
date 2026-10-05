# Samples GPU memory once per second until killed: adapter dedicated/shared, and (if -ProcName given)
# the dedicated/shared usage of that process. Output CSV: t,adapter_ded,adapter_shared,proc_ded,proc_shared (MB)
param([string]$Out, [string]$ProcName = "")
$ErrorActionPreference = "SilentlyContinue"
"t,adapter_ded,adapter_shared,proc_ded,proc_shared" | Out-File -Encoding ascii $Out
$luid = (Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage').CounterSamples | Sort-Object CookedValue -Descending | Select-Object -First 1
$inst = $luid.InstanceName
while ($true) {
    $pd = 0; $ps = 0
    $paths = @("\GPU Adapter Memory($inst)\Dedicated Usage", "\GPU Adapter Memory($inst)\Shared Usage")
    $p = if ($ProcName) { Get-Process -Name $ProcName -ErrorAction SilentlyContinue | Select-Object -First 1 } else { $null }
    if ($p) {
        $luidpart = $inst -replace '^luid_', ''
        $paths += "\GPU Process Memory(pid_$($p.Id)_luid_$luidpart)\Dedicated Usage"
        $paths += "\GPU Process Memory(pid_$($p.Id)_luid_$luidpart)\Shared Usage"
    }
    $s = (Get-Counter $paths).CounterSamples
    $ad = ($s | Where-Object { $_.Path -like '*adapter*dedicated*' }).CookedValue
    $as = ($s | Where-Object { $_.Path -like '*adapter*shared*' }).CookedValue
    $pdv = ($s | Where-Object { $_.Path -like '*process*dedicated*' }).CookedValue
    $psv = ($s | Where-Object { $_.Path -like '*process*shared*' }).CookedValue
    "{0},{1:F0},{2:F0},{3:F0},{4:F0}" -f [DateTime]::Now.ToString("HH:mm:ss"), ($ad/1MB), ($as/1MB), ($pdv/1MB), ($psv/1MB) | Out-File -Append -Encoding ascii $Out
    Start-Sleep -Milliseconds 700
}
