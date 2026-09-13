"""Artifact evidence database integration for Thamudic/NLP scanner results.

The artifact store extends the project's canonical SQLite approach with a normalized
research ledger for scans, translations, media provenance, annotations, voice actions,
and reproducible analysis metadata. It deliberately stores evidence and candidates
separately from scholarly claims; missing translations remain unavailable.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ArtifactDatabase:
    """Persistent SQLite store for scanner-derived artifact evidence."""

    def __init__(self, path: str | Path = "data/artifacts.sqlite"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA busy_timeout=30000")
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._initialize()

    def _initialize(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS artifacts (
                id TEXT PRIMARY KEY, title TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL DEFAULT '', media_type TEXT NOT NULL DEFAULT 'text',
                original_text TEXT NOT NULL DEFAULT '', source_language TEXT NOT NULL DEFAULT '',
                script_variant TEXT NOT NULL DEFAULT '', target_language TEXT NOT NULL DEFAULT '',
                transliteration TEXT, translation TEXT, translation_status TEXT,
                confidence TEXT, provider TEXT, provenance TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}', tags_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_artifacts_language ON artifacts(source_language);
            CREATE INDEX IF NOT EXISTS idx_artifacts_script ON artifacts(script_variant);
            CREATE INDEX IF NOT EXISTS idx_artifacts_status ON artifacts(translation_status);
            CREATE INDEX IF NOT EXISTS idx_artifacts_source ON artifacts(source);

            CREATE TABLE IF NOT EXISTS artifact_scans (
                id TEXT PRIMARY KEY, artifact_id TEXT NOT NULL, matched INTEGER NOT NULL DEFAULT 0,
                language TEXT, script_variant TEXT, confidence REAL, codepoints_json TEXT,
                scan_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL,
                FOREIGN KEY(artifact_id) REFERENCES artifacts(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_artifact_scans_artifact ON artifact_scans(artifact_id);

            CREATE TABLE IF NOT EXISTS artifact_translations (
                id TEXT PRIMARY KEY, artifact_id TEXT NOT NULL, source_form TEXT,
                target_language TEXT, transliteration TEXT, translation TEXT,
                status TEXT, confidence TEXT, provider TEXT, provenance TEXT,
                result_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL,
                FOREIGN KEY(artifact_id) REFERENCES artifacts(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_artifact_translations_artifact ON artifact_translations(artifact_id);

            CREATE TABLE IF NOT EXISTS artifact_media (
                id TEXT PRIMARY KEY, artifact_id TEXT NOT NULL, media_path TEXT,
                media_type TEXT, sha256 TEXT, extraction_method TEXT, extraction_status TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL,
                FOREIGN KEY(artifact_id) REFERENCES artifacts(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS artifact_annotations (
                id TEXT PRIMARY KEY, artifact_id TEXT NOT NULL, label TEXT,
                x REAL, y REAL, width REAL, height REAL, unicode_candidate TEXT,
                transliteration_candidate TEXT, confidence REAL, notes TEXT,
                created_at TEXT NOT NULL, FOREIGN KEY(artifact_id) REFERENCES artifacts(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS artifact_voice (
                id TEXT PRIMARY KEY, artifact_id TEXT NOT NULL, mode TEXT, language TEXT,
                backend TEXT, action TEXT, text_hash TEXT, metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL, FOREIGN KEY(artifact_id) REFERENCES artifacts(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS artifact_provenance (
                id TEXT PRIMARY KEY, artifact_id TEXT NOT NULL, source_name TEXT,
                source_url TEXT, rights TEXT, license TEXT, record_id TEXT, notes TEXT,
                created_at TEXT NOT NULL, FOREIGN KEY(artifact_id) REFERENCES artifacts(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS artifact_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT OR REPLACE INTO artifact_meta(key,value) VALUES ('schema_version','1');
            INSERT OR REPLACE INTO artifact_meta(key,value) VALUES ('evidence_policy','evidence-first; no fabricated translations');
            """
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "ArtifactDatabase":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @staticmethod
    def _json(value: Any, default: Any) -> str:
        return json.dumps(default if value is None else value, ensure_ascii=False)

    def create_artifact(self, record: dict[str, Any]) -> str:
        artifact_id = str(record.get("id") or uuid.uuid4().hex)
        now = _now()
        self.conn.execute(
            """INSERT OR REPLACE INTO artifacts
            (id,title,source,media_type,original_text,source_language,script_variant,target_language,
             transliteration,translation,translation_status,confidence,provider,provenance,metadata_json,tags_json,created_at,updated_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (artifact_id, str(record.get("title", "")), str(record.get("source", "")),
             str(record.get("media_type", "text")), str(record.get("original_text", record.get("text", ""))),
             str(record.get("source_language", record.get("language", ""))), str(record.get("script_variant", record.get("script", ""))),
             str(record.get("target_language", "")), record.get("transliteration"), record.get("translation"),
             record.get("translation_status"), str(record.get("confidence", "")), record.get("provider"), record.get("provenance"),
             self._json(record.get("metadata"), {}), self._json(record.get("tags"), []), now, now),
        )
        self.conn.commit()
        return artifact_id

    def add_scan(self, artifact_id: str, result: dict[str, Any]) -> str:
        scan_id = str(result.get("id") or uuid.uuid4().hex)
        self.conn.execute(
            "INSERT OR REPLACE INTO artifact_scans(id,artifact_id,matched,language,script_variant,confidence,codepoints_json,scan_json,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (scan_id, artifact_id, int(bool(result.get("matched"))), result.get("language"), result.get("script_variant"),
             float(result.get("confidence", 0) or 0), self._json(result.get("codepoints"), []), self._json(result, {}), _now()),
        )
        self.conn.commit()
        return scan_id

    def add_translation(self, artifact_id: str, result: dict[str, Any], source_form: str = "script") -> str:
        tid = str(uuid.uuid4().hex)
        self.conn.execute(
            "INSERT INTO artifact_translations(id,artifact_id,source_form,target_language,transliteration,translation,status,confidence,provider,provenance,result_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (tid, artifact_id, source_form, result.get("target_language"), result.get("transliteration"), result.get("translation"),
             result.get("translation_status", result.get("status")), str(result.get("confidence", "")), result.get("provider"), result.get("provenance"), self._json(result, {}), _now()),
        )
        self.conn.execute("UPDATE artifacts SET transliteration=?,translation=?,translation_status=?,confidence=?,provider=?,provenance=?,updated_at=? WHERE id=?",
                          (result.get("transliteration"), result.get("translation"), result.get("translation_status", result.get("status")), str(result.get("confidence", "")), result.get("provider"), result.get("provenance"), _now(), artifact_id))
        self.conn.commit()
        return tid

    def add_media(self, artifact_id: str, media: dict[str, Any]) -> str:
        mid = str(uuid.uuid4().hex)
        self.conn.execute(
            "INSERT INTO artifact_media(id,artifact_id,media_path,media_type,sha256,extraction_method,extraction_status,metadata_json,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (mid, artifact_id, media.get("media_path"), media.get("media_type"), media.get("sha256"), media.get("extraction_method"), media.get("extraction_status"), self._json(media.get("metadata"), {}), _now()),
        )
        self.conn.commit()
        return mid

    def add_annotation(self, artifact_id: str, annotation: dict[str, Any]) -> str:
        aid = str(annotation.get("id") or uuid.uuid4().hex)
        self.conn.execute(
            "INSERT OR REPLACE INTO artifact_annotations(id,artifact_id,label,x,y,width,height,unicode_candidate,transliteration_candidate,confidence,notes,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (aid, artifact_id, annotation.get("label"), annotation.get("x"), annotation.get("y"), annotation.get("width"), annotation.get("height"), annotation.get("unicode_candidate"), annotation.get("transliteration_candidate"), annotation.get("confidence"), annotation.get("notes"), _now()),
        )
        self.conn.commit()
        return aid

    def add_voice(self, artifact_id: str, voice: dict[str, Any]) -> str:
        vid = str(uuid.uuid4().hex)
        self.conn.execute(
            "INSERT INTO artifact_voice(id,artifact_id,mode,language,backend,action,text_hash,metadata_json,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (vid, artifact_id, voice.get("mode"), voice.get("language"), voice.get("backend"), voice.get("action"), voice.get("text_hash"), self._json(voice.get("metadata"), {}), _now()),
        )
        self.conn.commit()
        return vid

    def add_provenance(self, artifact_id: str, source: dict[str, Any]) -> str:
        pid = str(uuid.uuid4().hex)
        self.conn.execute(
            "INSERT INTO artifact_provenance(id,artifact_id,source_name,source_url,rights,license,record_id,notes,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (pid, artifact_id, source.get("source_name"), source.get("source_url"), source.get("rights"), source.get("license"), source.get("record_id"), source.get("notes"), _now()),
        )
        self.conn.commit()
        return pid

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, Any]:
        value = dict(row)
        for key in ("metadata_json", "tags_json"):
            raw = value.pop(key, None)
            if raw:
                try: value[key.removesuffix("_json")] = json.loads(raw)
                except json.JSONDecodeError: value[key.removesuffix("_json")] = raw
        return value

    def get(self, artifact_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM artifacts WHERE id=?", (artifact_id,)).fetchone()
        if not row: return None
        artifact = self._decode(row)
        artifact["scans"] = [dict(r) for r in self.conn.execute("SELECT * FROM artifact_scans WHERE artifact_id=? ORDER BY created_at", (artifact_id,))]
        artifact["translations"] = [dict(r) for r in self.conn.execute("SELECT * FROM artifact_translations WHERE artifact_id=? ORDER BY created_at", (artifact_id,))]
        artifact["media"] = [dict(r) for r in self.conn.execute("SELECT * FROM artifact_media WHERE artifact_id=? ORDER BY created_at", (artifact_id,))]
        artifact["annotations"] = [dict(r) for r in self.conn.execute("SELECT * FROM artifact_annotations WHERE artifact_id=? ORDER BY created_at", (artifact_id,))]
        artifact["voice"] = [dict(r) for r in self.conn.execute("SELECT * FROM artifact_voice WHERE artifact_id=? ORDER BY created_at", (artifact_id,))]
        artifact["provenance_records"] = [dict(r) for r in self.conn.execute("SELECT * FROM artifact_provenance WHERE artifact_id=? ORDER BY created_at", (artifact_id,))]
        return artifact

    def search(self, query: str = "", source_language: str = "", script_variant: str = "", limit: int = 100) -> list[dict[str, Any]]:
        sql = "SELECT * FROM artifacts WHERE 1=1"; args: list[Any] = []
        if query:
            q = f"%{query}%"; sql += " AND (title LIKE ? OR original_text LIKE ? OR transliteration LIKE ? OR translation LIKE ? OR source LIKE ?)"; args.extend([q] * 5)
        if source_language: sql += " AND source_language=?"; args.append(source_language)
        if script_variant: sql += " AND script_variant=?"; args.append(script_variant)
        sql += " ORDER BY updated_at DESC LIMIT ?"; args.append(max(1, min(int(limit), 1000)))
        return [self._decode(row) for row in self.conn.execute(sql, args)]

    def statistics(self) -> dict[str, Any]:
        tables = ("artifacts", "artifact_scans", "artifact_translations", "artifact_media", "artifact_annotations", "artifact_voice", "artifact_provenance")
        return {"database": str(self.path), "schema_version": "1", **{table: int(self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]) for table in tables}}
