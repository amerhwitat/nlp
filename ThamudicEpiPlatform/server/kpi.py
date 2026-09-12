from __future__ import annotations

import sqlite3
from typing import Any


def summary(conn: sqlite3.Connection, application: str = "ThamudicEpiPlatform") -> dict[str, Any]:
    def scalar(sql: str, params: tuple[Any, ...] = ()) -> int:
        row = conn.execute(sql, params).fetchone()
        return int(row[0] or 0)

    objects = scalar("SELECT COUNT(*) FROM objects")
    readings = scalar("SELECT COUNT(*) FROM readings")
    reviewed = scalar("SELECT COUNT(*) FROM readings WHERE status='reviewed'")
    translations = scalar("SELECT COUNT(*) FROM translation_results")
    pdf_imports = scalar("SELECT COUNT(*) FROM pdf_imports")
    pdf_exports = scalar("SELECT COUNT(*) FROM pdf_exports")
    ocr_jobs = scalar("SELECT COUNT(*) FROM ocr_jobs")
    ocr_completed = scalar("SELECT COUNT(*) FROM ocr_jobs WHERE status='completed'")
    ocr_errors = scalar("SELECT COUNT(*) FROM ocr_jobs WHERE status='error'")
    errors = scalar("SELECT COUNT(*) FROM pdf_imports WHERE status='error'") + ocr_errors
    avg_conf = conn.execute("SELECT AVG(confidence) FROM translation_results WHERE confidence IS NOT NULL").fetchone()[0]
    ocr_avg_conf = conn.execute("SELECT AVG(confidence) FROM ocr_jobs WHERE confidence IS NOT NULL").fetchone()[0]
    return {
        "schema_version": "1.1",
        "application": application,
        "metrics": {
            "objects_total": objects,
            "readings_total": readings,
            "reviewed_readings": reviewed,
            "translations_total": translations,
            "translation_confidence_mean": float(avg_conf or 0),
            "pdf_imports": pdf_imports,
            "pdf_exports": pdf_exports,
            "ocr_jobs": ocr_jobs,
            "ocr_completed": ocr_completed,
            "ocr_errors": ocr_errors,
            "ocr_confidence_mean": float(ocr_avg_conf or 0),
            "processing_errors": errors,
        },
    }


def languages(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT source_language, COUNT(*) AS translations, AVG(confidence) AS confidence "
        "FROM translation_results GROUP BY source_language ORDER BY translations DESC"
    ).fetchall()
    return [dict(r) for r in rows]
