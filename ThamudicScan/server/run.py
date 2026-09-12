from pathlib import Path
import os
import uvicorn

if __name__ == "__main__":
    host = os.getenv("THAMUDIC_HOST", "127.0.0.1")
    port = int(os.getenv("THAMUDIC_PORT", "8000"))
    uvicorn.run("ThamudicScan.server.main:app", host=host, port=port, reload=os.getenv("THAMUDIC_RELOAD", "0") == "1")
