@echo off
setlocal
set ROOT=%~dp0\..\..
set BUILD=%ROOT%\build\gpu
cmake -S "%ROOT%\gpu" -B "%BUILD%" -DCMAKE_BUILD_TYPE=Release -DAVRS_ENABLE_OPENGL=ON -DAVRS_ENABLE_DIRECTX12=ON
if errorlevel 1 exit /b %errorlevel%
cmake --build "%BUILD%" --config Release --parallel
exit /b %errorlevel%
