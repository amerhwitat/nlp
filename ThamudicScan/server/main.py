from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from .db import Database
from .exporter import export_results_csv, export_results_json
from .models import ScanRequest, ScanResponse, ScanResult, ScanSummary, ValidationRequest
from .progress import emit
from .scanner_adapter import scan_text, validate_text

APP_VERSION = "1.0.0"
DEFAULT_UPLOADS = {".txt", ".md", ".csv", ".json", ".xml", ".html", ".htm"}


def _db_path() -> Path:
    return Path(os.getenv("THAMUDIC_DB_PATH", str(Path(__file__).with_name("data") / "sessions.sqlite3")))


def _allowed_extensions() -> set[str]:
    raw = os.getenv("THAMUDIC_ALLOWED_EXTENSIONS", "")
    return {item.strip().lower() for item in raw.split(",") if item.strip()} or DEFAULT_UPLOADS


def _max_upload_bytes() -> int:
    return int(os.getenv("THAMUDIC_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))


def create_app() -> FastAPI:
    app = FastAPI(title="Thamudic Scanner API", version=APP_VERSION)
    origins = [origin.strip() for origin in os.getenv("THAMUDIC_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if origin.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["*"])
    db = Database(_db_path())
    app.state.db = db

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "thamudic-scanner", "version": APP_VERSION}

    @app.post("/validate")
    def validate(request: ValidationRequest):
        return validate_text(request.text)

    def execute_scan(text: str, keywords: list[str], source: str) -> ScanResponse:
        session_id = db.create_session()
        db.update_session(session_id, status="running")
        emit(db, session_id, event_type="started", status="running", progress=0, processed_count=0, match_count=0, source=source, message="Scan started")
        result_data = scan_text(text, keywords)
        results: list[ScanResult] = []
        processed = 1
        matches = 1 if result_data["matched"] else 0
        if result_data["matched"]:
            result = {
                "session_id": session_id,
                "source": source,
                "text": result_data["text"],
                "transliteration": result_data["transliteration"],
                "confidence": result_data["confidence"],
                "language": result_data["language"],
                "script_variant": result_data["script_variant"],
                "codepoints": result_data["codepoints"],
            }
            result["id"] = db.save_result(session_id, result)
            results.append(ScanResult(**result))
        db.update_session(session_id, status="completed", processed_count=processed, match_count=matches)
        emit(db, session_id, event_type="completed", status="completed", progress=100, processed_count=processed, match_count=matches, source=source, message="Scan completed")
        return ScanResponse(session_id=session_id, results=results, summary=ScanSummary(processed_count=processed, match_count=matches, status="completed"))

    @app.post("/scan", response_model=ScanResponse)
    async def scan(request: ScanRequest):
        return await asyncio.to_thread(execute_scan, request.text, request.keywords, request.source)

    async def _read_upload(upload: UploadFile) -> bytes:
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix not in _allowed_extensions():
            raise HTTPException(status_code=415, detail="Unsupported upload type")
        maximum = _max_upload_bytes()
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > maximum:
                raise HTTPException(status_code=413, detail="Uploaded file exceeds configured size limit")
            chunks.append(chunk)
        return b"".join(chunks)

    @app.post("/scan_file", response_model=ScanResponse)
    async def scan_file(file: UploadFile = File(...)):
        data = await _read_upload(file)
        suffix = Path(file.filename or "").suffix.lower()
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise HTTPException(status_code=422, detail="File must contain valid UTF-8 text") from exc
        source = Path(file.filename or "uploaded-file").name
        return await asyncio.to_thread(execute_scan, text, [], source)

    @app.get("/sessions/{session_id}")
    def get_session(session_id: str):
        session = db.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"session": session, "results": db.list_results(session_id), "events": db.list_events(session_id)}

    @app.get("/sessions/{session_id}/events")
    async def events(session_id: str):
        if not db.get_session(session_id):
            raise HTTPException(status_code=404, detail="Session not found")

        async def stream():
            last = 0
            idle_rounds = 0
            while idle_rounds < 50:
                batch = db.list_events(session_id, after=last)
                if batch:
                    for event in batch:
                        last = event["sequence"]
                        yield f"id: {last}\\ndata: {json.dumps(event, ensure_ascii=False)}\\n\\n"
                        if event["status"] in {"completed", "failed", "cancelled"}:
                            return
                    idle_rounds = 0
                else:
                    idle_rounds += 1
                await asyncio.sleep(0.1)

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/export/{session_id}")
    def export_session(session_id: str, format: str = "csv"):
        if not db.get_session(session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        results = db.list_results(session_id)
        if format.lower() == "csv":
            return Response(export_results_csv(results), media_type="text/csv; charset=utf-8", headers={"Content-Disposition": f'attachment; filename="thamudic-{session_id}.csv"'})
        if format.lower() == "json":
            return Response(export_results_json(results), media_type="application/json; charset=utf-8", headers={"Content-Disposition": f'attachment; filename="thamudic-{session_id}.json"'})
        raise HTTPException(status_code=400, detail="format must be csv or json")

    return app


app = create_app()
