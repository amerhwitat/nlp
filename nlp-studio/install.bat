@echo off
rem Install NLP / Ancient Script Studio dependencies into a local virtualenv.
cd /d "%~dp0"

echo Creating virtual environment (.venv)...
python -m venv .venv
if errorlevel 1 goto :fail

call .venv\Scripts\activate.bat

echo Installing Python dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 goto :fail

echo.
echo Done. Dependencies installed into nlp-studio\.venv
echo Run the app with: run.bat
exit /b 0

:fail
echo Installation failed.
exit /b 1
