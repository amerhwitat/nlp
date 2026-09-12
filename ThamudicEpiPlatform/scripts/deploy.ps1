$ErrorActionPreference='Stop'
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r server\requirements.txt
New-Item -ItemType Directory -Force data | Out-Null
.\.venv\Scripts\python.exe -m uvicorn server.app:app --host ($env:HOST ?? '0.0.0.0') --port ([int]($env:PORT ?? '8010'))
