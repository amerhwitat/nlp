$ErrorActionPreference='Stop'
$Root=Resolve-Path "$PSScriptRoot\..";Set-Location $Root
$strict=$env:STRICT_TOOLCHAINS -eq '1'
function Run-Optional($name,[scriptblock]$action){if(Get-Command $name -ErrorAction SilentlyContinue){& $action}else{Write-Warning "SKIP: $name not installed";if($strict){exit 2}}}
Run-Optional cmake {cmake -S cpp -B cpp/build;cmake --build cpp/build}
Run-Optional dotnet {dotnet build csharp\ThamudicOcr.csproj}
Run-Optional mvn {mvn -q -f java\pom.xml package}
Run-Optional go {go build -o go\ocr_scan.exe .\go}
Run-Optional cargo {cargo build --manifest-path rust\Cargo.toml}
Run-Optional node {node --check javascript\ocr_scan.mjs}
Run-Optional npm {Push-Location typescript;npm install;npm run build;Pop-Location}
