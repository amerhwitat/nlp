@echo off
setlocal EnableExtensions
set "ROOT=%~dp0..\.."
for %%I in ("%ROOT%") do set "ROOT=%%~fI"
set "PROFILE=%~1"
if not defined PROFILE set "PROFILE=developer"
where pwsh >nul 2>&1 && (pwsh -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\scripts\bootstrap\install-dependencies.ps1" -Profile "%PROFILE%" & exit /b %ERRORLEVEL%)
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\scripts\bootstrap\install-dependencies.ps1" -Profile "%PROFILE%"
exit /b %ERRORLEVEL%
