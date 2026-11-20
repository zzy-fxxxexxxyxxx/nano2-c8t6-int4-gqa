# Flash nano2 firmware to STM32F103C8 via OpenOCD + ST-Link V2 (SWD).
# ASCII only: PowerShell 5.1 reads .ps1 as ANSI.
param(
    [string]$Elf = '',
    [int]$Speed = 950
)
$ErrorActionPreference = 'Stop'
$root    = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$ocdRoot = Join-Path $root 'tools\openocd\xpack-openocd-0.12.0-7'
$ocd     = Join-Path $ocdRoot 'bin\openocd.exe'
$scripts = Join-Path $ocdRoot 'openocd\scripts'
if (-not $Elf) { $Elf = Join-Path $root 'fw\build\Debug\nano2.elf' }

if (-not (Test-Path $Elf)) { throw "firmware not found: $Elf" }
Write-Host ("firmware: {0} ({1:N0} bytes)" -f $Elf, (Get-Item $Elf).Length)

$elfPosix = $Elf -replace '\\', '/'
$cmd = "adapter speed $Speed; init; reset halt; program `"$elfPosix`" verify; reset run; shutdown"
& $ocd -s $scripts -f interface/stlink.cfg -f target/stm32f1x.cfg -c $cmd
exit $LASTEXITCODE
