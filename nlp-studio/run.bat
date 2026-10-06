@echo off
rem Run the NLP / Ancient Script Studio backend + single-page app.
cd /d "%~dp0"

if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
if "%PORT%"=="" set PORT=8000

echo Starting NLP / Ancient Script Studio on http://localhost:%PORT%
python -m uvicorn backend.main:app --host 0.0.0.0 --port %PORT%
