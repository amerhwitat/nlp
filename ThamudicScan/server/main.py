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
from .scanner_adapter import (alphabet_languages, alphabet_profile, alphabet_variations, scan_source_language_text, translation_directions_for, translation_modes, scan_text, translate_text, validate_text, translate_ancient_text, all_translation_directions, script_summary, voice_speak, voice_backends, voice_commands, speech_recognition)
from python.thamudic.script_summary import export_script_summary

APP_VERSION = "1.5.0"
DEFAULT_UPLOADS = {".txt", ".md", ".csv", ".json", ".xml", ".html", ".htm"}


def _db_path() -> Path: return Path(os.getenv("THAMUDIC_DB_PATH", str(Path(__file__).with_name("data") / "sessions.sqlite3")))
def _allowed_extensions() -> set[str]:
    raw = os.getenv("THAMUDIC_ALLOWED_EXTENSIONS", "")
    return {item.strip().lower() for item in raw.split(",") if item.strip()} or DEFAULT_UPLOADS

def _max_upload_bytes() -> int: return int(os.getenv("THAMUDIC_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))


def create_app() -> FastAPI:
    app = FastAPI(title="Thamudic Scanner API", version=APP_VERSION)
    origins = [o.strip() for o in os.getenv("THAMUDIC_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["*"])
    db = Database(_db_path()); app.state.db = db

    @app.get("/health")
    def health(): return {"status": "ok", "service": "thamudic-scanner", "version": APP_VERSION}
    @app.get("/alphabet-languages")
    def alphabet_language_endpoint(): return {"languages": list(alphabet_languages())}
    @app.get("/alphabet-languages/{language}")
    def alphabet_profile_endpoint(language: str):
        try: return {"profile": alphabet_profile(language), "variations": list(alphabet_variations(language)), "translation_modes": list(translation_modes(language)), "translation_directions": translation_directions_for(language)}
        except ValueError as exc: raise HTTPException(status_code=404, detail=str(exc)) from exc
    @app.get("/translation-matrix")
    def translation_matrix_endpoint(): return {"languages": all_translation_directions()}
    @app.get("/script-summary/{language}")
    def script_summary_endpoint(language: str):
        try: return script_summary(language)
        except ValueError as exc: raise HTTPException(status_code=404, detail=str(exc)) from exc
    @app.get("/script-summary/{language}/export")
    def script_summary_export_endpoint(language: str, format: str = "json"):
        try: body, media_type, filename = export_script_summary(language, format)
        except ValueError as exc: raise HTTPException(status_code=400, detail=str(exc)) from exc
        return Response(body, media_type=media_type, headers={"Content-Disposition": f'attachment; filename="{filename}"'})
    @app.get("/voice/capabilities")
    def voice_capabilities_endpoint(): return {"tts_backends": voice_backends(), "commands": voice_commands(), "speech_recognition": speech_recognition()}
    @app.post("/voice/speak")
    def voice_speak_endpoint(request: dict):
        text, language, mode = str(request.get("text", "")), str(request.get("language", "en")), str(request.get("mode", "translation"))
        try: return voice_speak(text, language, mode)
        except ValueError as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc
    @app.post("/validate")
    def validate(request: ValidationRequest): return validate_text(request.text)
    @app.post("/translate")
    def translate_endpoint(request: dict):
        text, script, target_language = str(request.get("text", "")), str(request.get("script", "Dadanitic")), str(request.get("target_language", "en"))
        if not text.strip(): raise HTTPException(status_code=422, detail="text is required")
        if target_language.casefold().split("-")[0] not in {"en", "ar"}: raise HTTPException(status_code=400, detail="supported target languages: en, ar in the deterministic baseline")
        return translate_text(text, script=script, target_language=target_language)
    @app.post("/translate_ancient")
    def translate_ancient_endpoint(request: dict):
        text, language, target, source_form = str(request.get("text", "")), str(request.get("source_language", "")), str(request.get("target_language", "")), str(request.get("source_form", "script"))
        if not text.strip() or not language.strip() or not target.strip(): raise HTTPException(status_code=422, detail="text, source_language and target_language are required")
        try: return translate_ancient_text(text, language, target, source_form=source_form)
        except ValueError as exc: raise HTTPException(status_code=400, detail=str(exc)) from exc
    @app.post("/scan_language")
    def scan_language_endpoint(request: dict):
        text, language = str(request.get("text", "")), request.get("language")
        if not text.strip(): raise HTTPException(status_code=422, detail="text is required")
        try: return scan_source_language_text(text, language=str(language) if language else None)
        except ValueError as exc: raise HTTPException(status_code=400, detail=str(exc)) from exc

    def execute_scan(text: str, keywords: list[str], source: str) -> ScanResponse:
        session_id = db.create_session(); db.update_session(session_id, status="running")
        emit(db, session_id, event_type="started", status="running", progress=0, processed_count=0, match_count=0, source=source, message="Scan started")
        result_data = scan_text(text, keywords); results: list[ScanResult] = []; processed = 1; matches = 1 if result_data["matched"] else 0
        if result_data["matched"]:
            result = {"session_id": session_id, "source": source, "text": result_data["text"], "transliteration": result_data["transliteration"], "confidence": result_data["confidence"], "language": result_data["language"], "script_variant": result_data["script_variant"], "codepoints": result_data["codepoints"]}
            result["id"] = db.save_result(session_id, result); results.append(ScanResult(**result))
        db.update_session(session_id, status="completed", processed_count=processed, match_count=matches)
        emit(db, session_id, event_type="completed", status="completed", progress=100, processed_count=processed, match_count=matches, source=source, message="Scan completed")
        return ScanResponse(session_id=session_id, results=results, summary=ScanSummary(processed_count=processed, match_count=matches, status="completed"))
    @app.post("/scan", response_model=ScanResponse)
    async def scan(request: ScanRequest): return await asyncio.to_thread(execute_scan, request.text, request.keywords, request.source)
    async def _read_upload(upload: UploadFile) -> bytes:
        if Path(upload.filename or "").suffix.lower() not in _allowed_extensions(): raise HTTPException(status_code=415, detail="Unsupported upload type")
        maximum, chunks, total = _max_upload_bytes(), [], 0
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk: break
            total += len(chunk)
            if total > maximum: raise HTTPException(status_code=413, detail="Uploaded file exceeds configured size limit")
            chunks.append(chunk)
        return b"".join(chunks)
    @app.post("/scan_file", response_model=ScanResponse)
    async def scan_file(file: UploadFile = File(...)):
        data = await _read_upload(file)
        try: text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc: raise HTTPException(status_code=422, detail="File must contain valid UTF-8 text") from exc
        return await asyncio.to_thread(execute_scan, text, [], Path(file.filename or "uploaded-file").name)
    @app.get("/sessions/{session_id}")
    def get_session(session_id: str):
        session = db.get_session(session_id)
        if not session: raise HTTPException(status_code=404, detail="Session not found")
        return {"session": session, "results": db.list_results(session_id), "events": db.list_events(session_id)}
    @app.get("/sessions/{session_id}/events")
    async def events(session_id: str):
        if not db.get_session(session_id): raise HTTPException(status_code=404, detail="Session not found")
        async def stream():
            last, idle = 0, 0
            while idle < 50:
                batch = db.list_events(session_id, after=last)
                if batch:
                    for event in batch:
                        last = event["sequence"]; yield f"id: {last}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
                        if event["status"] in {"completed", "failed", "cancelled"}: return
                    idle = 0
                else: idle += 1
                await asyncio.sleep(0.1)
        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    @app.get("/export/{session_id}")
    def export_session(session_id: str, format: str = "csv"):
        if not db.get_session(session_id): raise HTTPException(status_code=404, detail="Session not found")
        results = db.list_results(session_id)
        if format.lower() == "csv": return Response(export_results_csv(results), media_type="text/csv; charset=utf-8", headers={"Content-Disposition": f'attachment; filename="thamudic-{session_id}.csv"'})
        if format.lower() == "json": return Response(export_results_json(results), media_type="application/json; charset=utf-8", headers={"Content-Disposition": f'attachment; filename="thamudic-{session_id}.json"'})
        raise HTTPException(status_code=400, detail="format must be csv or json")
    return app

app = create_app()
