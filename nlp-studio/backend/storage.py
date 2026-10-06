"""Durable storage for uploaded media and OCR/translation results.

Uploaded files are stored under a randomly generated name (UUID) and their
original filename is kept only as display metadata. Uploads are never executed,
imported, or interpreted as code -- they are only ever handed to image/PDF
libraries as data.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_upload(filename: str, content: bytes) -> dict[str, Any]:
    """Persist an uploaded file with a safe, generated name and return its record."""
    config.ensure_directories()
    suffix = Path(filename or "").suffix.casefold() or ".bin"
    stored_id = uuid.uuid4().hex
    stored_name = f"{stored_id}{suffix}"
    destination = config.UPLOAD_DIR / stored_name
    destination.write_bytes(content)
    return {
        "id": stored_id,
        "original_name": Path(filename).name,
        "stored_name": stored_name,
        "path": str(destination),
        "size_bytes": len(content),
        "uploaded_at": _now(),
    }


def save_result(result_id: str, payload: dict[str, Any]) -> Path:
    config.ensure_directories()
    destination = config.RESULT_DIR / f"{result_id}.json"
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def load_result(result_id: str) -> dict[str, Any] | None:
    destination = config.RESULT_DIR / f"{result_id}.json"
    if not destination.exists():
        return None
    return json.loads(destination.read_text(encoding="utf-8"))


def list_results() -> list[dict[str, Any]]:
    config.ensure_directories()
    results: list[dict[str, Any]] = []
    for path in sorted(config.RESULT_DIR.glob("*.json")):
        try:
            results.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return results
