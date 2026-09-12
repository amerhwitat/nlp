[CmdletBinding()]
param([ValidateSet('Check','Init')][string]$Mode='Check',[string]$DatabaseUrl=$env:NLP_DATABASE_URL)
$ErrorActionPreference='Stop'
$Root=(Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $Root
if(-not $DatabaseUrl){ $DatabaseUrl=$env:DATABASE_URL }
if($Mode -eq 'Check'){
  if(Get-Command sqlite3 -ErrorAction SilentlyContinue){ sqlite3 --version }
  if(Get-Command psql -ErrorAction SilentlyContinue){ psql --version }
  Write-Host 'Database prerequisite check completed.'
  exit 0
}
if(-not (Test-Path 'db/sql')){ throw 'db/sql directory is missing.' }
New-Item -ItemType Directory -Force -Path 'artifacts/database' | Out-Null
if($DatabaseUrl -and (Get-Command psql -ErrorAction SilentlyContinue)){
  Get-ChildItem db/sql -Filter '*.sql' | Sort-Object Name | ForEach-Object { psql $DatabaseUrl -v ON_ERROR_STOP=1 -f $_.FullName }
} elseif(Get-Command sqlite3 -ErrorAction SilentlyContinue -ErrorAction SilentlyContinue){
  $db=Join-Path $Root 'artifacts/database/nlp.sqlite'
  Get-ChildItem db/sql -Filter '*.sql' | Sort-Object Name | ForEach-Object { sqlite3 $db ".read '$($_.FullName.Replace("'","''"))'" }
} else { throw 'No supported database client found. Install PostgreSQL client or SQLite.' }
Write-Host 'Database initialization completed.'
