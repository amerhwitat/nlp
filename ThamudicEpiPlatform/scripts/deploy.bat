@echo off
setlocal
cd /d "%~dp0.."
call scripts\check-dependencies.bat
call scripts\build-all-languages.bat
if not exist data mkdir data
.venv\Scripts\python.exe -m uvicorn server.app:app --host 0.0.0.0 --port 8010
