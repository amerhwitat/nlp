[CmdletBinding()]
param(
  [ValidateSet('minimal','developer','research','server','ci')]
  [string]$Profile = 'developer',
  [switch]$SkipDatabase
)
$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $RepoRoot

function Require-Command([string]$Name) {
  if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
    throw "Required command '$Name' was not found on PATH. Install it or run the platform bootstrap first."
  }
  & $Name --version 2>$null | Select-Object -First 1
}

Write-Host "NLP bootstrap: profile=$Profile root=$RepoRoot"
Require-Command git
Require-Command node
Require-Command npm
Require-Command python

if (Test-Path 'web/package.json') {
  Push-Location 'web'
  if (Test-Path 'package-lock.json') { npm ci } else { npm install }
  Pop-Location
}

if (-not (Test-Path '.venv')) { python -m venv .venv }
$Python = Join-Path $RepoRoot '.venv\Scripts\python.exe'
if (Test-Path 'python/requirements.txt') { & $Python -m pip install --upgrade pip; & $Python -m pip install -r 'python/requirements.txt' }

if (-not $SkipDatabase -and $Profile -in @('developer','research','server','ci')) {
  if (Test-Path 'scripts/database/init-database.ps1') { & (Join-Path $RepoRoot 'scripts/database/init-database.ps1') -Mode Check }
}

Write-Host 'Bootstrap completed.'
