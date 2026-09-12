$ErrorActionPreference='Stop'
Set-Location (Resolve-Path "$PSScriptRoot\..")
& "$PWD\scripts\check-dependencies.ps1"
$HostValue = if($env:HOST){$env:HOST}else{'0.0.0.0'}
$PortValue = if($env:PORT){[int]$env:PORT}else{8010}
New-Item -ItemType Directory -Force data | Out-Null
& .\.venv\Scripts\python.exe -m uvicorn server.app:app --host $HostValue --port $PortValue
