$ErrorActionPreference='Stop';$r=Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path);& (Join-Path $r 'build-tools/build.ps1') --only python --onefile @args;exit $LASTEXITCODE
