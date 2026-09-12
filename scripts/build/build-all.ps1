[CmdletBinding()]
param([ValidateSet('Debug','Release')][string]$Configuration='Release',[switch]$SkipNative,[switch]$SkipDotnet,[switch]$SkipPython,[switch]$SkipWasm,[switch]$SkipDatabase)
$ErrorActionPreference='Stop'
$Root=(Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $Root
function Invoke-Step([string]$Name,[scriptblock]$Action){ Write-Host "== $Name =="; & $Action; if($LASTEXITCODE -ne 0){throw "$Name failed with exit code $LASTEXITCODE"} }
if(Test-Path 'web/package.json'){ Invoke-Step 'Web build' { Push-Location web; npm run build; Pop-Location } }
if(-not $SkipWasm -and (Test-Path 'scripts/build/build-wasm.ps1')){ Invoke-Step 'WASM build' { & (Join-Path $Root 'scripts/build/build-wasm.ps1') -Configuration $Configuration } }
if(-not $SkipNative -and (Test-Path 'scripts/build/build-native.ps1')){ Invoke-Step 'Native build' { & (Join-Path $Root 'scripts/build/build-native.ps1') -Configuration $Configuration } }
if(-not $SkipDotnet -and (Test-Path 'dotnet')){ Invoke-Step '.NET build' { dotnet build 'dotnet' -c $Configuration --nologo } }
if(-not $SkipPython -and (Test-Path 'python')){ Invoke-Step 'Python validation' { python -m compileall -q python } }
if(-not $SkipDatabase -and (Test-Path 'scripts/database/init-database.ps1')){ Invoke-Step 'Database validation' { & (Join-Path $Root 'scripts/database/init-database.ps1') -Mode Check } }
Write-Host "Build-all completed: $Configuration"
