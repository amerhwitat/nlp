$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$vendor = Join-Path $root 'vendor'
New-Item -ItemType Directory -Force -Path $vendor | Out-Null
$sae = Join-Path $vendor 'sae'
$vamiga = Join-Path $vendor 'vamigaweb'
if (-not (Test-Path (Join-Path $sae '.git'))) { git clone https://github.com/naTmeg/ScriptedAmigaEmulator.git $sae } else { git -C $sae pull --ff-only }
if (-not (Test-Path (Join-Path $vamiga '.git'))) { git clone https://github.com/vAmigaWeb/vAmigaWeb.git $vamiga } else { git -C $vamiga pull --ff-only }
Write-Host 'Upstream Amiga sources are available under Amiga/vendor/. Review each upstream license before redistribution.'
