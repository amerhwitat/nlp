"""Supplementary artifact persistence endpoints for scanner feature telemetry."""
from __future__ import annotations

import hashlib
import os
from typing import Any
from fastapi import APIRouter, HTTPException
from python.thamudic.artifacts_db import ArtifactDatabase
from .scanner_adapter import voice_speak, voice_backends, voice_commands, speech_recognition, script_summary

router = APIRouter(prefix="/artifacts", tags=["artifact-features"])


def _db():
    return ArtifactDatabase(os.getenv("THAMUDIC_ARTIFACT_DB_PATH", "data/artifacts.sqlite"))


def _require(db: ArtifactDatabase, artifact_id: str) -> dict[str, Any]:
    item = db.get(artifact_id)
    if not item:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return item


@router.post("/{artifact_id}/annotation")
def add_annotation(artifact_id: str, annotation: dict[str, Any]):
    with _db() as db:
        _require(db, artifact_id); annotation_id = db.add_annotation(artifact_id, annotation)
        return {"artifact_id": artifact_id, "annotation_id": annotation_id}


@router.post("/{artifact_id}/provenance")
def add_provenance(artifact_id: str, source: dict[str, Any]):
    with _db() as db:
        _require(db, artifact_id); provenance_id = db.add_provenance(artifact_id, source)
        return {"artifact_id": artifact_id, "provenance_id": provenance_id}


@router.post("/{artifact_id}/voice")
def add_voice_action(artifact_id: str, request: dict[str, Any]):
    text = str(request.get("text", "")); mode = str(request.get("mode", "translation")); language = str(request.get("language", "en"))
    if not text.strip(): raise HTTPException(status_code=422, detail="text is required")
    result = voice_speak(text, language, mode)
    with _db() as db:
        _require(db, artifact_id)
        voice_id = db.add_voice(artifact_id, {"mode": mode, "language": language, "backend": result.get("backend"), "action": result.get("status", "speak"), "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(), "metadata": result})
    return {"artifact_id": artifact_id, "voice_id": voice_id, "result": result}


@router.get("/voice/capabilities")
def artifact_voice_capabilities():
    return {"tts_backends": voice_backends(), "commands": voice_commands(), "speech_recognition": speech_recognition()}


@router.get("/script/{language}")
def artifact_script_metadata(language: str):
    try: return script_summary(language)
    except ValueError as exc: raise HTTPException(status_code=404, detail=str(exc)) from exc
