from __future__ import annotations

from .db import Database


def emit(db: Database, session_id: str, *, event_type: str, status: str, progress: int, processed_count: int, match_count: int, source: str | None = None, message: str | None = None) -> dict:
    return db.add_event(session_id, {
        "type": event_type,
        "status": status,
        "progress": max(0, min(100, progress)),
        "processed_count": processed_count,
        "match_count": match_count,
        "source": source,
        "message": message,
    })
