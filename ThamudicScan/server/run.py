from __future__ import annotations

import os
import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ThamudicScan.server.main import app  # noqa: E402


if __name__ == "__main__":
    host = os.getenv("THAMUDIC_HOST", "127.0.0.1")
    port = int(os.getenv("THAMUDIC_PORT", "8000"))
    uvicorn.run(app, host=host, port=port, reload=os.getenv("THAMUDIC_RELOAD", "0") == "1")
