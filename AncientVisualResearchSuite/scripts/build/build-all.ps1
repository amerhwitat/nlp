$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Write-Host "Ancient Visual Research Suite build: $Root"
if (Get-Command cmake -ErrorAction SilentlyContinue) {
  cmake -S "$Root/core/cpp" -B "$Root/build/cpp" -DCMAKE_BUILD_TYPE=Release
  cmake --build "$Root/build/cpp" --config Release
} else { Write-Warning 'CMake not installed; skipping C++ build.' }
if (Get-Command python -ErrorAction SilentlyContinue) {
  python -m compileall "$Root/core/python"
}
if (Test-Path "$Root/web/package.json") {
  if (Get-Command npm -ErrorAction SilentlyContinue) { Push-Location "$Root/web"; npm ci; npm run build; Pop-Location }
}
Write-Host 'Reference builds complete where toolchains are installed.'
