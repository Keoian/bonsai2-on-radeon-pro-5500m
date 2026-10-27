<#
.SYNOPSIS
  Save the server's current chat (what it has already read) to disk, or load it back after a restart.

.EXAMPLE
  .\chat-cache.ps1 save             # before stopping the server: writes slots\chat.bin
  .\chat-cache.ps1 restore          # after starting it again, before the next message
  .\chat-cache.ps1 save work        # a named file, slots\work.bin
  .\chat-cache.ps1 list

  Without a restore, the first message after a restart reads the whole chat again (about 25-35 tokens/s on the
  5500M, so 15+ minutes for a 30k-token chat). A restore takes seconds. The file holds the KV cache, the Gated
  DeltaNet checkpoints and the MTP draft state, so it is large (about 1-2 GB for a long chat), and it only fits a
  server started with the same model, KV cache types and -Mtp setting.
#>
[CmdletBinding(PositionalBinding = $true)]
param(
    [Parameter(Mandatory = $true)] [ValidateSet("save", "restore", "list")] [string] $Action,
    [string] $Name = "chat",
    [int]    $Port = 8080,
    [int]    $Slot = 0,
    [string] $SlotDir = (Join-Path $PSScriptRoot "slots")
)

$ErrorActionPreference = "Stop"

if ($Action -eq "list") {
    Get-ChildItem $SlotDir -Filter *.bin -ErrorAction SilentlyContinue |
        Select-Object @{ n = "Name"; e = { $_.BaseName } }, @{ n = "MB"; e = { [Math]::Round($_.Length / 1MB) } }, LastWriteTime |
        Format-Table -AutoSize
    return
}

if ($Action -eq "restore" -and -not (Test-Path (Join-Path $SlotDir "$Name.bin"))) { throw "no saved chat '$Name' in $SlotDir (.\chat-cache.ps1 list)" }

$body = [System.Text.Encoding]::UTF8.GetBytes((@{ filename = "$Name.bin" } | ConvertTo-Json))
try {
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/slots/$($Slot)?action=$Action" -Method Post -ContentType "application/json" -Body $body -TimeoutSec 600
} catch {
    $msg = $_.ErrorDetails.Message
    if ($msg -match "not supported|slot-save-path") { $msg += " (start the server with start-server.ps1, which sets --slot-save-path)" }
    throw "$Action failed: $($_.Exception.Message) $msg"
}
if ($Action -eq "save") {
    "saved {0} tokens to {1} ({2:N0} MB, {3:N1} s)" -f $r.n_saved, (Join-Path $SlotDir "$Name.bin"), ($r.n_written / 1MB), ($r.timings.save_ms / 1000)
} else {
    "restored {0} tokens from {1} ({2:N0} MB, {3:N1} s)" -f $r.n_restored, (Join-Path $SlotDir "$Name.bin"), ($r.n_read / 1MB), ($r.timings.restore_ms / 1000)
}
