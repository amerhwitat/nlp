"""SQLite database layer for the Thamudic / Ancient Object Research system.

SQLite is the canonical local database. JSON and SQL dumps are interchange
formats/backups; no proprietary database server is required. The API is kept
small so the same database can be used by the Tkinter GUI, CLI tools and
future Chimera II OS/Web UI adapters.
"""
from __future__ import annotations

import csv
import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Iterable

from historical_periods import get_period

SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS objects (
 id TEXT PRIMARY KEY, title TEXT NOT NULL, period_key TEXT, period_name TEXT,
 object_type TEXT, culture TEXT, script_key TEXT, language TEXT,
 date_start TEXT, date_end TEXT, site TEXT, region TEXT, country TEXT,
 current_location TEXT, material TEXT, technique TEXT, description TEXT,
 transliteration TEXT, translation_ar TEXT, translation_en TEXT,
 source_name TEXT, source_url TEXT, source_record_id TEXT, image_url TEXT,
 image_page_url TEXT, image_iiif TEXT, image_local_path TEXT,
 license TEXT, rights_notes TEXT, creator TEXT, provenance TEXT,
 bibliography TEXT, subjects TEXT, latitude REAL, longitude REAL,
 confidence REAL, reviewer TEXT, competing_readings TEXT, tags TEXT,
 created_at TEXT DEFAULT CURRENT_TIMESTAMP, updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_objects_period ON objects(period_key);
CREATE INDEX IF NOT EXISTS idx_objects_script ON objects(script_key);
CREATE INDEX IF NOT EXISTS idx_objects_type ON objects(object_type);
CREATE INDEX IF NOT EXISTS idx_objects_source ON objects(source_name);
CREATE INDEX IF NOT EXISTS idx_objects_country ON objects(country);
CREATE INDEX IF NOT EXISTS idx_objects_site ON objects(site);
CREATE UNIQUE INDEX IF NOT EXISTS idx_objects_source_record
 ON objects(source_name, source_record_id)
 WHERE source_name IS NOT NULL AND source_record_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS annotations (
 id TEXT PRIMARY KEY, object_id TEXT NOT NULL, label TEXT, x REAL, y REAL,
 width REAL, height REAL, unicode_candidate TEXT, transliteration_candidate TEXT,
 confidence REAL, reviewer TEXT, notes TEXT,
 FOREIGN KEY(object_id) REFERENCES objects(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_annotations_object ON annotations(object_id);

CREATE TABLE IF NOT EXISTS sources (
 id TEXT PRIMARY KEY, name TEXT UNIQUE, homepage TEXT, api_url TEXT,
 rights_policy TEXT, image_policy TEXT, enabled INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS database_meta (key TEXT PRIMARY KEY, value TEXT);
INSERT OR REPLACE INTO database_meta(key,value) VALUES ('schema_version','2');
INSERT OR REPLACE INTO database_meta(key,value) VALUES ('storage_format','SQLite + JSON/SQL interchange');
"""

LIST_FIELDS = {"subjects", "tags", "competing_readings", "bibliography"}


class ObjectDatabase:
    """Persistent local evidence database backed by one portable SQLite file."""

    def __init__(self, path: str | Path = "ancient_objects.sqlite"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA busy_timeout=30000")
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "ObjectDatabase":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @staticmethod
    def _encode(value: Any) -> Any:
        return json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value

    @staticmethod
    def _decode_row(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        for key in LIST_FIELDS:
            value = result.get(key)
            if value:
                try:
                    result[key] = json.loads(value)
                except (TypeError, json.JSONDecodeError):
                    pass
        return result

    def _object_columns(self) -> set[str]:
        return {row[1] for row in self.conn.execute("PRAGMA table_info(objects)")}

    def _prepare_object(self, record: dict[str, Any]) -> dict[str, Any]:
        data = dict(record)
        data["id"] = data.get("id") or str(uuid.uuid4())
        if data.get("period_key"):
            try:
                data["period_name"] = get_period(data["period_key"])["name"]
            except KeyError:
                pass
        for key in LIST_FIELDS:
            if key in data:
                data[key] = self._encode(data[key])
        return {k: v for k, v in data.items() if k in self._object_columns()}

    def add_object(self, record: dict[str, Any]) -> str:
        """Insert or replace one object and return its stable record ID."""
        data = self._prepare_object(record)
        fields = list(data)
        self.conn.execute(
            f"INSERT OR REPLACE INTO objects ({','.join(fields)}) VALUES ({','.join('?' for _ in fields)})",
            [data[k] for k in fields],
        )
        self.conn.commit()
        return str(data["id"])

    def import_records(self, records: Iterable[dict[str, Any]]) -> int:
        """Import records atomically; return the number imported."""
        count = 0
        try:
            self.conn.execute("BEGIN")
            for record in records:
                data = self._prepare_object(record)
                fields = list(data)
                self.conn.execute(
                    f"INSERT OR REPLACE INTO objects ({','.join(fields)}) VALUES ({','.join('?' for _ in fields)})",
                    [data[k] for k in fields],
                )
                count += 1
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        return count

    def get_object(self, object_id: str) -> dict[str, Any]:
        row = self.conn.execute("SELECT * FROM objects WHERE id=?", (object_id,)).fetchone()
        if not row:
            raise KeyError(object_id)
        return self._decode_row(row)

    def list_objects(self, query: str = "", period_key: str | None = None,
                     script_key: str | None = None, object_type: str | None = None,
                     country: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM objects WHERE 1=1"
        args: list[Any] = []
        if query:
            sql += " AND (title LIKE ? OR description LIKE ? OR tags LIKE ? OR subjects LIKE ? OR transliteration LIKE ? OR translation_en LIKE ? OR translation_ar LIKE ?)"
            q = f"%{query}%"
            args.extend([q] * 7)
        for column, value in (("period_key", period_key), ("script_key", script_key),
                              ("object_type", object_type), ("country", country)):
            if value:
                sql += f" AND {column}=?"
                args.append(value)
        sql += " ORDER BY updated_at DESC, title COLLATE NOCASE"
        if limit is not None:
            sql += " LIMIT ?"
            args.append(max(0, int(limit)))
        return [self._decode_row(row) for row in self.conn.execute(sql, args)]

    def add_annotation(self, object_id: str, annotation: dict[str, Any]) -> str:
        aid = annotation.get("id") or str(uuid.uuid4())
        data = dict(annotation); data.update({"id": aid, "object_id": object_id})
        fields = list(data)
        self.conn.execute(
            f"INSERT OR REPLACE INTO annotations ({','.join(fields)}) VALUES ({','.join('?' for _ in fields)})",
            [data[k] for k in fields],
        )
        self.conn.commit()
        return str(aid)

    def add_source(self, source: dict[str, Any]) -> str:
        sid = source.get("id") or str(uuid.uuid4())
        data = dict(source); data["id"] = sid
        fields = list(data)
        self.conn.execute(
            f"INSERT OR REPLACE INTO sources ({','.join(fields)}) VALUES ({','.join('?' for _ in fields)})",
            [data[k] for k in fields],
        )
        self.conn.commit()
        return str(sid)

    def export_csv(self, path: str | Path) -> int:
        rows = self.list_objects()
        fields = list(rows[0]) if rows else ["id", "title"]
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            writer.writeheader(); writer.writerows(rows)
        return len(rows)

    def export_json(self, path: str | Path) -> int:
        rows = self.list_objects()
        Path(path).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        return len(rows)

    def import_json(self, path: str | Path) -> int:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("JSON database import must contain a list of records")
        return self.import_records(data)

    def export_sql(self, path: str | Path) -> int:
        target = Path(path)
        with target.open("w", encoding="utf-8") as fh:
            for line in self.conn.iterdump():
                fh.write(line + "\n")
        return target.stat().st_size

    def backup(self, path: str | Path) -> Path:
        """Create a consistent SQLite backup using SQLite's backup API."""
        target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(target) as dest:
            self.conn.backup(dest)
        return target

    def statistics(self) -> dict[str, Any]:
        def count(table: str) -> int:
            return int(self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        return {
            "database": str(self.path),
            "objects": count("objects"),
            "annotations": count("annotations"),
            "sources": count("sources"),
            "schema_version": self.conn.execute("SELECT value FROM database_meta WHERE key='schema_version'").fetchone()[0],
        }
