@echo off
setlocal EnableExtensions
set "ROOT=%~dp0..\.."
for %%I in ("%ROOT%") do set "ROOT=%%~fI"
set "MODE=%~1"
if not defined MODE set "MODE=Check"
where pwsh >nul 2>&1 && (pwsh -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\scripts\database\init-database.ps1" -Mode "%MODE%" & exit /b %ERRORLEVEL%)
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\scripts\database\init-database.ps1" -Mode "%MODE%"
exit /b %ERRORLEVEL%
