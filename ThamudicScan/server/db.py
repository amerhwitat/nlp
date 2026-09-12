from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self):
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    processed_count INTEGER NOT NULL DEFAULT 0,
                    match_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS results (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    source TEXT NOT NULL,
                    text TEXT NOT NULL,
                    transliteration TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    language TEXT NOT NULL,
                    script_variant TEXT NOT NULL,
                    codepoints TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_results_session ON results(session_id);
                CREATE TABLE IF NOT EXISTS events (
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    sequence INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (session_id, sequence)
                );
                """
            )

    def create_session(self) -> str:
        session_id = uuid.uuid4().hex
        now = _now()
        with self._lock, self._connect() as db:
            db.execute(
                "INSERT INTO sessions(id,status,created_at,updated_at) VALUES(?,?,?,?)",
                (session_id, "created", now, now),
            )
        return session_id

    def update_session(self, session_id: str, *, status: str | None = None, processed_count: int | None = None, match_count: int | None = None):
        fields = ["updated_at = ?"]
        values: list[object] = [_now()]
        if status is not None:
            fields.append("status = ?")
            values.append(status)
        if processed_count is not None:
            fields.append("processed_count = ?")
            values.append(processed_count)
        if match_count is not None:
            fields.append("match_count = ?")
            values.append(match_count)
        values.append(session_id)
        with self._lock, self._connect() as db:
            db.execute(f"UPDATE sessions SET {', '.join(fields)} WHERE id = ?", values)

    def save_result(self, session_id: str, result: dict) -> str:
        result_id = result.get("id") or uuid.uuid4().hex
        now = _now()
        with self._lock, self._connect() as db:
            db.execute(
                "INSERT INTO results(id,session_id,source,text,transliteration,confidence,language,script_variant,codepoints,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (result_id, session_id, result.get("source", ""), result.get("text", ""), result.get("transliteration", ""), float(result.get("confidence", 0)), result.get("language", "Old North Arabian"), result.get("script_variant", "Dadanitic"), json.dumps(result.get("codepoints", []), ensure_ascii=False), now),
            )
        return result_id

    def get_session(self, session_id: str) -> dict | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return dict(row) if row else None

    def list_results(self, session_id: str) -> list[dict]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM results WHERE session_id = ? ORDER BY created_at, id", (session_id,)).fetchall()
        return [self._result(row) for row in rows]

    @staticmethod
    def _result(row) -> dict:
        value = dict(row)
        value["codepoints"] = json.loads(value["codepoints"])
        return value

    def add_event(self, session_id: str, payload: dict) -> dict:
        with self._lock, self._connect() as db:
            row = db.execute("SELECT COALESCE(MAX(sequence), 0) + 1 AS next_sequence FROM events WHERE session_id = ?", (session_id,)).fetchone()
            sequence = int(row["next_sequence"])
            event = dict(payload)
            event["sequence"] = sequence
            db.execute("INSERT INTO events(session_id,sequence,payload,created_at) VALUES(?,?,?,?)", (session_id, sequence, json.dumps(event, ensure_ascii=False), _now()))
        return event

    def list_events(self, session_id: str, after: int = 0) -> list[dict]:
        with self._connect() as db:
            rows = db.execute("SELECT payload FROM events WHERE session_id = ? AND sequence > ? ORDER BY sequence", (session_id, after)).fetchall()
        return [json.loads(row["payload"]) for row in rows]
