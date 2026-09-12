@echo off
setlocal
cd /d "%~dp0.."
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r server\requirements.txt
if "%INSTALL_KRAKEN%"=="1" .venv\Scripts\python.exe -m pip install "kraken>=7,<8"
where tesseract >nul 2>nul && echo Tesseract binary: available || echo Tesseract binary: not found (optional)
.venv\Scripts\python.exe -c "import PIL,numpy,cv2,fastapi,pypdf,reportlab; print('Python OCR/API dependencies: OK')"
