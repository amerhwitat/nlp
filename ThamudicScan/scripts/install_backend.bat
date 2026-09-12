@echo off
setlocal
cd /d "%~dp0..\.."
py -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r ThamudicScan\server\requirements.txt
endlocal
