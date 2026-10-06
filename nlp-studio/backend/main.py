"""FastAPI service for NLP / Ancient Script Studio.

Endpoints:
    GET  /api/health                       service + capability status
    POST /api/upload                       image/PDF upload -> OCR -> save result
    POST /api/translate                    text -> transliteration + translation
    GET  /api/languages                    supported scripts / languages / targets
    GET  /api/script-summary/{language}    script metadata (optionally exported)
    GET  /api/history                      durable translation history
    GET  /api/export/history               export history as json/jsonl/txt/pdf
    GET  /api/export/result/{result_id}    export a saved result as json/md/txt/pdf
    GET  /                                the single-page app

Uploads are bounded by size and type, validated by extension and magic bytes,
stored under generated names, and never executed or imported as code.
"""
from __future__ import annotations

import json
import unicodedata
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import config, storage
from .ocr_service import ocr_availability, ocr_file
from .translation_service import translate_text

from python.thamudic import (
    build_script_summary,
    export_script_summary,
    supported_alphabet_languages,
    translation_directions,
)
from python.thamudic.old_north_arabian import BY_CHARACTER, VARIANT_FORMS, is_old_north_arabian
from python.thamudic.source_language_scanner import scan_source_language
from python.thamudic.translation_log import export_records, read_records, verify_records

app = FastAPI(title=config.APP_NAME, version=config.APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# --- helpers -----------------------------------------------------------------

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _detect_script(text: str) -> dict[str, Any]:
    """Scan source-language profiles and additionally detect Old North Arabian.

    ``scan_source_language`` covers the generic classical Unicode profiles but not
    the Old North Arabian block (U+10A80..U+10A9F), so we augment its result with
    the ONA registry from ``old_north_arabian``.
    """
    scan = scan_source_language(text) if text else {
        "text": text, "detected_languages": [], "counts": {},
        "characters": [], "matched_character_count": 0,
    }
    ona_chars = []
    for index, ch in enumerate(text):
        if is_old_north_arabian(ch):
            info = BY_CHARACTER.get(ch)
            ona_chars.append({
                "index": index,
                "character": ch,
                "codepoint": f"U+{ord(ch):04X}",
                "decimal": ord(ch),
                "name": info["name"] if info else unicodedata.name(ch, "UNNAMED"),
                "transliteration": info["transliteration"] if info else None,
                "matches": ["ancient-north-arabian"],
            })
    if ona_chars:
        counts = scan.setdefault("counts", {})
        counts["ancient-north-arabian"] = len(ona_chars)
        merged = scan.get("characters", []) + ona_chars
        scan["characters"] = sorted(merged, key=lambda item: item["index"])
        seen: list[str] = []
        for language in scan.get("detected_languages", []) + ["ancient-north-arabian"]:
            if language not in seen:
                seen.append(language)
        scan["detected_languages"] = seen
        scan["matched_character_count"] = len(scan["characters"])
    return scan


def _flatten(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _render_payload(payload: dict[str, Any], fmt: str, base: str) -> tuple[bytes, str, str]:
    fmt = fmt.casefold()
    title = str(payload.get("title") or payload.get("name") or base)
    if fmt == "json":
        return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"), "application/json; charset=utf-8", f"{base}.json"
    if fmt == "txt":
        lines = [f"{title} — report", ""]
        for key, value in payload.items():
            lines.append(f"{key}: {_flatten(value)}")
        return ("\n".join(lines) + "\n").encode("utf-8"), "text/plain; charset=utf-8", f"{base}.txt"
    if fmt == "md":
        lines = [f"# {title} — report", ""]
        for key, value in payload.items():
            lines.append(f"- **{key.replace('_', ' ').title()}**: {_flatten(value)}")
        return ("\n".join(lines) + "\n").encode("utf-8"), "text/markdown; charset=utf-8", f"{base}.md"
    if fmt == "pdf":
        from python.thamudic.pdf_export import report_pdf_bytes
        return report_pdf_bytes(payload), "application/pdf", f"{base}.pdf"
    raise ValueError("format must be json, md, txt, or pdf")


def _export_response(body: bytes, media_type: str, filename: str) -> Response:
    return Response(
        content=body,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _validate_upload(filename: str, content: bytes) -> str | None:
    suffix = ("." + filename.rsplit(".", 1)[-1].casefold()) if "." in filename else ""
    if suffix not in config.ALLOWED_SUFFIXES:
        return f"unsupported file type '{suffix or '(none)'}'; allowed: {', '.join(sorted(config.ALLOWED_SUFFIXES))}"
    if len(content) > config.MAX_UPLOAD_BYTES:
        return f"file too large (max {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB)"
    if suffix == ".pdf":
        if not content.lstrip()[:4] == b"%PDF":
            return "file is not a valid PDF"
    else:
        try:
            from PIL import Image
            import io
            with Image.open(io.BytesIO(content)) as image:
                image.verify()
        except Exception:
            return "file is not a valid image"
    return None


# --- API ---------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict[str, Any]:
    config.ensure_directories()
    return {
        "status": "ok",
        "app": config.APP_NAME,
        "version": config.APP_VERSION,
        "ocr": ocr_availability(),
        "translation_provider_enabled": config.provider_enabled(),
        "target_languages": list(config.SUPPORTED_TARGET_LANGUAGES),
    }


@app.post("/api/upload")
async def upload(
    file: UploadFile = File(...),
    script: str = Form("Dadanitic"),
    target_language: str = Form("en"),
    source_language: str = Form("ancient-north-arabian"),
    ocr_languages: str = Form(""),
) -> JSONResponse:
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="a file is required")
    content = await file.read(config.MAX_UPLOAD_BYTES + 1)
    error = _validate_upload(file.filename or "", content)
    if error:
        raise HTTPException(status_code=400, detail=error)

    try:
        build_script_summary(source_language)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    upload_record = storage.save_upload(file.filename or "", content)
    ocr = ocr_file(upload_record["path"], [x.strip() for x in ocr_languages.split(",") if x.strip()] or None)

    text = ocr.get("text") or ""
    scan = _detect_script(text)
    translation = translate_text(
        text,
        script=script,
        source_language=source_language,
        target_language=target_language,
        request_metadata={"endpoint": "upload", "result_flow": "upload -> OCR -> scan -> transliterate -> translate"},
    ) if text else None

    summary = build_script_summary(source_language)
    payload = {
        "title": config.APP_NAME,
        "result_id": upload_record["id"],
        "created_at": upload_record["uploaded_at"],
        "script": script,
        "source_language": source_language,
        "target_language": target_language,
        "upload": {k: v for k, v in upload_record.items() if k != "path"},
        "ocr": ocr,
        "scan": scan,
        "translation": translation,
        "script_information": summary,
        "workflow": "upload -> OCR -> scan -> transliterate -> translate",
    }
    storage.save_result(upload_record["id"], payload)
    return JSONResponse(status_code=200, content=payload)


@app.post("/api/translate")
async def translate_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="request body must be JSON") from exc
    text = str(body.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    script = str(body.get("script") or "Dadanitic")
    source_language = str(body.get("source_language") or "ancient-north-arabian")
    target_language = str(body.get("target_language") or "en")

    try:
        summary = build_script_summary(source_language)
        result = translate_text(
            text,
            script=script,
            source_language=source_language,
            target_language=target_language,
            request_metadata={"endpoint": "translate"},
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    scan = _detect_script(text)
    result_id = uuid.uuid4().hex
    payload = {
        "title": config.APP_NAME,
        "result_id": result_id,
        "created_at": _utcnow(),
        "script": script,
        "source_language": source_language,
        "target_language": target_language,
        "scan": scan,
        "translation": result,
        "script_information": summary,
        "workflow": "text -> scan -> transliterate -> translate",
    }
    storage.save_result(result_id, payload)
    return JSONResponse(status_code=200, content=payload)


@app.get("/api/languages")
def languages() -> dict[str, Any]:
    profiles = []
    for language_id in supported_alphabet_languages():
        summary = build_script_summary(language_id)
        profiles.append({
            "id": language_id,
            "name": summary["name"],
            "original_script": summary["original_script"],
            "writing_direction": summary["writing_direction"],
            "translation_directions": translation_directions(language_id),
        })
    return {
        "source_languages": profiles,
        "ona_scripts": list(VARIANT_FORMS),
        "target_languages": list(config.SUPPORTED_TARGET_LANGUAGES),
    }


@app.get("/api/script-summary/{language}")
def script_summary(language: str, format: str = "json") -> Any:
    try:
        if format.casefold() == "json":
            return build_script_summary(language)
        body, media_type, filename = export_script_summary(language, format)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _export_response(body.encode("utf-8") if isinstance(body, str) else body, media_type, filename)


@app.get("/api/history")
def history() -> dict[str, Any]:
    records = read_records(config.DATA_DIR / "translations.jsonl")
    return {"records": records, "count": len(records), "integrity": verify_records(records)}


@app.get("/api/export/history")
def export_history(format: str = "json") -> Response:
    try:
        body, media_type, filename = export_records(format, config.DATA_DIR / "translations.jsonl")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _export_response(body.encode("utf-8") if isinstance(body, str) else body, media_type, filename)


@app.get("/api/export/result/{result_id}")
def export_result(result_id: str, format: str = "json") -> Response:
    payload = storage.load_result(result_id)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"result '{result_id}' not found")
    try:
        body, media_type, filename = _render_payload(payload, format, f"nlp-studio-{result_id}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _export_response(body, media_type, filename)


# Serve the single-page app (after API routes so they take precedence).
config.ensure_directories()
if config.WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(config.WEB_DIR), html=True), name="web")
