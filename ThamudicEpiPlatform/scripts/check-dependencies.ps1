$ErrorActionPreference='Stop'
$Root=Resolve-Path "$PSScriptRoot\..";Set-Location $Root
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r server\requirements.txt
if($env:INSTALL_KRAKEN -eq '1'){.\.venv\Scripts\python.exe -m pip install 'kraken>=7,<8'}
if(Get-Command tesseract -ErrorAction SilentlyContinue){Write-Host 'Tesseract binary: available'}else{Write-Warning 'Tesseract binary not found (optional; install via winget/choco if required)'}
\.venv\Scripts\python.exe -c "import PIL,numpy,cv2,pytesseract,fastapi,pypdf,reportlab; print('Python OCR/API dependencies: OK')"
\.venv\Scripts\python.exe -c "import importlib.util; print('kraken: available' if importlib.util.find_spec('kraken') else 'kraken: optional/not installed')"
