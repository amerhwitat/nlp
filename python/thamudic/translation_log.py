"""Durable, provenance-first translation history.

Logs are append-only JSON Lines plus JSON/TXT/PDF exports. Sensitive credentials
are never accepted as log fields. Translation content is recorded together with
script metadata, direction, provider, confidence, provenance, timestamps, and a
deterministic record hash for integrity checks.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_log_path() -> Path:
    return Path(os.getenv("THAMUDIC_TRANSLATION_LOG", "translation_logs/translations.jsonl"))


@dataclass
class TranslationLogRecord:
    timestamp: str
    source: str
    source_language: str
    source_form: str
    target_language: str
    target_form: str | None
    transliteration: str | None
    translation: str | None
    status: str
    confidence: float
    provider: str
    provenance: str | None = None
    script_metadata: dict[str, Any] = field(default_factory=dict)
    request_metadata: dict[str, Any] = field(default_factory=dict)
    record_hash: str = ""

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["record_hash"] = self.record_hash or compute_record_hash(data)
        return data


def compute_record_hash(data: dict[str, Any]) -> str:
    canonical = dict(data)
    canonical.pop("record_hash", None)
    payload = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_record(*, source: str, source_language: str, source_form: str, target_language: str,
                target_form: str | None = None, transliteration: str | None = None,
                translation: str | None = None, status: str = "unknown", confidence: float = 0.0,
                provider: str = "none", provenance: str | None = None,
                script_metadata: dict[str, Any] | None = None,
                request_metadata: dict[str, Any] | None = None) -> TranslationLogRecord:
    record = TranslationLogRecord(
        timestamp=_utc(), source=source, source_language=source_language,
        source_form=source_form, target_language=target_language, target_form=target_form,
        transliteration=transliteration, translation=translation, status=status,
        confidence=float(confidence), provider=provider, provenance=provenance,
        script_metadata=script_metadata or {}, request_metadata=request_metadata or {},
    )
    record.record_hash = compute_record_hash(asdict(record))
    return record


def append_record(record: TranslationLogRecord, path: str | Path | None = None) -> Path:
    target = Path(path) if path else default_log_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.as_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    return target


def read_records(path: str | Path | None = None) -> list[dict[str, Any]]:
    target = Path(path) if path else default_log_path()
    if not target.exists():
        return []
    records: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid translation log JSON at line {line_number}") from exc
    return records


def verify_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    total = valid = 0
    invalid_hashes: list[str] = []
    for record in records:
        total += 1
        expected = compute_record_hash(record)
        if record.get("record_hash") == expected:
            valid += 1
        else:
            invalid_hashes.append(str(record.get("record_hash", "")))
    return {"total": total, "valid": valid, "invalid": total - valid, "invalid_hashes": invalid_hashes}


def export_records(format: str = "json", path: str | Path | None = None) -> tuple[str, str, str]:
    records = read_records(path)
    fmt = format.casefold()
    if fmt == "json":
        return json.dumps(records, ensure_ascii=False, indent=2) + "\n", "application/json; charset=utf-8", "translation-log.json"
    if fmt == "jsonl":
        return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records), "application/x-ndjson; charset=utf-8", "translation-log.jsonl"
    if fmt == "txt":
        blocks: list[str] = []
        for index, record in enumerate(records, 1):
            blocks.append("=" * 72); blocks.append(f"Translation record {index}")
            for key, value in record.items():
                blocks.append(f"{key}: {json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value}")
        return "\n".join(blocks) + ("\n" if blocks else ""), "text/plain; charset=utf-8", "translation-log.txt"
    if fmt == "pdf":
        from .pdf_export import records_pdf_bytes
        return records_pdf_bytes(records), "application/pdf", "translation-log.pdf"
    raise ValueError("format must be json, jsonl, txt, or pdf")
