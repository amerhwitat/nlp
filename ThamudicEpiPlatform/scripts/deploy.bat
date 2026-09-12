@echo off
setlocal
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r server\requirements.txt
if not exist data mkdir data
.venv\Scripts\python.exe -m uvicorn server.app:app --host 0.0.0.0 --port 8010
