"""Web API for the unified Thamudic/NLP artifact evidence database."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response

from python.thamudic.artifacts_db import ArtifactDatabase
from python.thamudic.media_pipeline import extract_media_text
from .scanner_adapter import scan_source_language_text, scan_text, translate_ancient_text, translate_text

router = APIRouter(prefix="/artifacts", tags=["artifacts"])


def _artifact_db() -> ArtifactDatabase:
    return ArtifactDatabase(os.getenv("THAMUDIC_ARTIFACT_DB_PATH", "data/artifacts.sqlite"))


def _ingest(record: dict[str, Any]) -> dict[str, Any]:
    with _artifact_db() as db:
        artifact_id = db.create_artifact(record)
        if record.get("scan") is not None:
            db.add_scan(artifact_id, record["scan"])
        if record.get("translation") is not None:
            db.add_translation(artifact_id, record["translation"], record.get("source_form", "script"))
        if record.get("media") is not None:
            db.add_media(artifact_id, record["media"])
        if isinstance(record.get("provenance"), dict):
            db.add_provenance(artifact_id, record["provenance"])
        for annotation in record.get("annotations", []):
            db.add_annotation(artifact_id, annotation)
        for voice in record.get("voice", []):
            db.add_voice(artifact_id, voice)
        return db.get(artifact_id) or {"id": artifact_id}


@router.get("/stats")
def artifact_stats():
    with _artifact_db() as db:
        return db.statistics()


@router.get("/search")
def artifact_search(q: str = "", source_language: str = "", script_variant: str = "", limit: int = 100):
    with _artifact_db() as db:
        return {"results": db.search(q, source_language, script_variant, limit)}


@router.get("/export/json")
def artifact_export_json(q: str = "", source_language: str = "", script_variant: str = "", limit: int = 1000):
    with _artifact_db() as db:
        data = db.search(q, source_language, script_variant, limit)
    return Response(json.dumps(data, ensure_ascii=False, indent=2), media_type="application/json; charset=utf-8", headers={"Content-Disposition": "attachment; filename=artifacts.json"})


@router.post("/ingest")
def artifact_ingest(record: dict[str, Any]):
    if not str(record.get("original_text", record.get("text", ""))).strip():
        raise HTTPException(status_code=422, detail="original_text or text is required")
    return _ingest(record)


@router.post("/analyze")
def artifact_analyze(request: dict[str, Any]):
    text = str(request.get("text", "")); script = str(request.get("script", "Dadanitic")); target = str(request.get("target_language", "en")); source_language = str(request.get("source_language", script))
    if not text.strip(): raise HTTPException(status_code=422, detail="text is required")
    scan = scan_text(text, list(request.get("keywords", [])))
    language_scan = scan_source_language_text(text, language=source_language or None)
    translation = translate_text(text, script=script, target_language=target)
    return _ingest({"title": str(request.get("title", "NLP/Thamudic analysis")), "source": str(request.get("source", "web")), "media_type": "text", "original_text": text, "source_language": source_language, "script_variant": script, "target_language": target, "scan": {**scan, "language_scan": language_scan}, "translation": translation, "source_form": "script", "metadata": {"pipeline": "web-artifact-analyze", "language_scan": language_scan}, "tags": request.get("tags", [])})


@router.post("/analyze-ancient")
def artifact_analyze_ancient(request: dict[str, Any]):
    text = str(request.get("text", "")); source_language = str(request.get("source_language", "")); target = str(request.get("target_language", "en")); source_form = str(request.get("source_form", "script"))
    if not text.strip() or not source_language.strip() or not target.strip(): raise HTTPException(status_code=422, detail="text, source_language and target_language are required")
    translation = translate_ancient_text(text, source_language, target, source_form=source_form, request_metadata={"api": "/artifacts/analyze-ancient"})
    scan = scan_source_language_text(text, language=source_language)
    return _ingest({"title": str(request.get("title", "Ancient language artifact")), "source": str(request.get("source", "web")), "media_type": "text", "original_text": text, "source_language": source_language, "script_variant": source_language, "target_language": target, "scan": {"matched": bool(scan.get("matches")), "language": source_language, "script_variant": source_language, "confidence": 0, "codepoints": [], "language_scan": scan}, "translation": translation, "source_form": source_form, "metadata": {"pipeline": "web-artifact-ancient", "language_scan": scan}, "tags": request.get("tags", [])})


@router.post("/analyze-upload")
async def artifact_analyze_upload(file: UploadFile = File(...), script: str = "Dadanitic", target_language: str = "en"):
    suffix = Path(file.filename or "artifact").suffix.lower()
    allowed = {".txt", ".md", ".csv", ".json", ".xml", ".html", ".htm", ".pdf", ".png", ".jpg", ".jpeg", ".webp"}
    if suffix not in allowed: raise HTTPException(status_code=415, detail="Unsupported artifact type")
    data = await file.read()
    if len(data) > int(os.getenv("THAMUDIC_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024))): raise HTTPException(status_code=413, detail="Uploaded artifact exceeds configured size limit")
    digest = hashlib.sha256(data).hexdigest()
    with tempfile.TemporaryDirectory(prefix="thamudic-artifact-") as tmp:
        path = Path(tmp) / (Path(file.filename or "artifact").name or "artifact")
        path.write_bytes(data)
        try:
            text, media = extract_media_text(path)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Media extraction failed: {exc}") from exc
    if not text.strip(): raise HTTPException(status_code=422, detail="No text could be extracted from artifact")
    scan = scan_text(text, []); language_scan = scan_source_language_text(text, language=script); translation = translate_text(text, script=script, target_language=target_language)
    return _ingest({"title": Path(file.filename or "artifact").name, "source": "web-upload", "media_type": suffix.lstrip("."), "original_text": text, "source_language": script, "script_variant": script, "target_language": target_language, "scan": {**scan, "language_scan": language_scan}, "translation": translation, "media": {"media_path": file.filename, "media_type": suffix.lstrip("."), "sha256": digest, "extraction_method": media.get("provider", "media_pipeline"), "extraction_status": "ok", "metadata": media}, "metadata": {"pipeline": "web-artifact-upload", "original_filename": file.filename, "sha256": digest, "media": media}})


@router.get("/{artifact_id}")
def artifact_get(artifact_id: str):
    with _artifact_db() as db: result = db.get(artifact_id)
    if not result: raise HTTPException(status_code=404, detail="Artifact not found")
    return result
