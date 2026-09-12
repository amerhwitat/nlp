@echo off
setlocal
set ROOT=%~dp0..
call "%ROOT%build-tools\build.bat" --only python --onefile %*
exit /b %ERRORLEVEL%
