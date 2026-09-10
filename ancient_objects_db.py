"""SQLite evidence database for inscriptions and historical objects."""
from __future__ import annotations
import csv, json, sqlite3, uuid
from pathlib import Path
from typing import Any
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
 image_iiif TEXT, image_local_path TEXT, license TEXT, rights_notes TEXT,
 creator TEXT, provenance TEXT, bibliography TEXT, subjects TEXT,
 latitude REAL, longitude REAL, confidence REAL, reviewer TEXT,
 competing_readings TEXT, tags TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
 updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_objects_period ON objects(period_key);
CREATE INDEX IF NOT EXISTS idx_objects_script ON objects(script_key);
CREATE INDEX IF NOT EXISTS idx_objects_type ON objects(object_type);
CREATE INDEX IF NOT EXISTS idx_objects_source ON objects(source_name);
CREATE TABLE IF NOT EXISTS annotations (
 id TEXT PRIMARY KEY, object_id TEXT NOT NULL, label TEXT, x REAL, y REAL,
 width REAL, height REAL, unicode_candidate TEXT, transliteration_candidate TEXT,
 confidence REAL, reviewer TEXT, notes TEXT,
 FOREIGN KEY(object_id) REFERENCES objects(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS sources (
 id TEXT PRIMARY KEY, name TEXT UNIQUE, homepage TEXT, api_url TEXT,
 rights_policy TEXT, image_policy TEXT, enabled INTEGER DEFAULT 1
);
"""

class ObjectDatabase:
    def __init__(self, path="ancient_objects.sqlite"):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        self.conn=sqlite3.connect(self.path); self.conn.row_factory=sqlite3.Row
        self.conn.executescript(SCHEMA); self.conn.commit()

    def close(self): self.conn.close()

    def add_object(self, record:dict[str,Any])->str:
        data=dict(record); oid=data.get("id") or str(uuid.uuid4()); data["id"]=oid
        if data.get("period_key"):
            try: data["period_name"]=get_period(data["period_key"])["name"]
            except KeyError: pass
        for key in ("subjects","tags","competing_readings","bibliography"):
            if isinstance(data.get(key),(list,dict)): data[key]=json.dumps(data[key],ensure_ascii=False)
        cols={row[1] for row in self.conn.execute("PRAGMA table_info(objects)")}; data={k:v for k,v in data.items() if k in cols}
        fields=",".join(data); marks=",".join("?" for _ in data)
        self.conn.execute(f"INSERT OR REPLACE INTO objects ({fields}) VALUES ({marks})",list(data.values())); self.conn.commit(); return oid

    def get_object(self, object_id):
        row=self.conn.execute("SELECT * FROM objects WHERE id=?",(object_id,)).fetchone()
        if not row: raise KeyError(object_id)
        return dict(row)

    def list_objects(self, query="", period_key=None, script_key=None, object_type=None):
        sql="SELECT * FROM objects WHERE 1=1"; args=[]
        if query:
            sql += " AND (title LIKE ? OR description LIKE ? OR tags LIKE ? OR subjects LIKE ?)"; q=f"%{query}%"; args += [q,q,q,q]
        if period_key: sql += " AND period_key=?"; args.append(period_key)
        if script_key: sql += " AND script_key=?"; args.append(script_key)
        if object_type: sql += " AND object_type=?"; args.append(object_type)
        sql += " ORDER BY updated_at DESC, title"
        return [dict(r) for r in self.conn.execute(sql,args)]

    def add_annotation(self, object_id, annotation):
        aid=annotation.get("id") or str(uuid.uuid4()); data=dict(annotation); data.update({"id":aid,"object_id":object_id})
        fields=list(data); values=[data[k] for k in fields]
        self.conn.execute(f"INSERT OR REPLACE INTO annotations ({','.join(fields)}) VALUES ({','.join('?' for _ in values)})",values); self.conn.commit(); return aid

    def export_csv(self,path):
        rows=self.list_objects(); fields=list(rows[0]) if rows else ["id","title"]
        with open(path,"w",newline="",encoding="utf-8-sig") as fh:
            writer=csv.DictWriter(fh,fieldnames=fields); writer.writeheader(); writer.writerows(rows)

    def export_json(self,path):
        Path(path).write_text(json.dumps(self.list_objects(),ensure_ascii=False,indent=2),encoding="utf-8")
