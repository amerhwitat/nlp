@echo off
setlocal
set ROOT=%~dp0..\..
if exist "%ProgramFiles%\CMake\bin\cmake.exe" (
  cmake -S "%ROOT%\core\cpp" -B "%ROOT%\build\cpp"
  if errorlevel 1 exit /b 1
  cmake --build "%ROOT%\build\cpp" --config Release
)
where python >nul 2>nul && python -m compileall "%ROOT%\core\python"
if exist "%ROOT%\web\package.json" (
  where npm >nul 2>nul && (cd /d "%ROOT%\web" && npm ci && npm run build)
)
echo AVRS reference build finished where toolchains are installed.
endlocal
