<#
.SYNOPSIS
  Bonsai 2 27B with a 16k context: fits on the 8 GB 5500M next to a normal desktop.
  All other arguments pass through to start-server.ps1 (e.g. -Lan, -NoVision, extra llama-server flags).
#>
& (Join-Path $PSScriptRoot "start-server.ps1") -Ctx 16384 @args
exit $LASTEXITCODE
