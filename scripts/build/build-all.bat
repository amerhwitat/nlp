@echo off
setlocal EnableExtensions
set "ROOT=%~dp0..\.."
for %%I in ("%ROOT%") do set "ROOT=%%~fI"
set "CONFIGURATION=%~1"
if not defined CONFIGURATION set "CONFIGURATION=Release"
where pwsh >nul 2>&1 && (pwsh -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\scripts\build\build-all.ps1" -Configuration "%CONFIGURATION%" & exit /b %ERRORLEVEL%)
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\scripts\build\build-all.ps1" -Configuration "%CONFIGURATION%"
exit /b %ERRORLEVEL%
