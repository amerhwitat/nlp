@echo off
setlocal
cd /d "%~dp0..\.."
call .venv\Scripts\activate.bat
python ThamudicScan\server\run.py
endlocal
