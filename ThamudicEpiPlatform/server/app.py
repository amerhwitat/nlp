from __future__ import annotations

import csv
import io
import os
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
DB = Path(os.getenv('THAMUDIC_PLATFORM_DB', ROOT / 'data' / 'thamudic_platform.sqlite'))
DB.parent.mkdir(parents=True, exist_ok=True)

app = FastAPI(title='Thamudic Cross-Language Epigraphy API', version='1.0.0')
origins = [x.strip() for x in os.getenv('CORS_ORIGINS', 'http://localhost:5173,http://127.0.0.1:5173').split(',') if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=['*'], allow_headers=['*'])

class ObjectIn(BaseModel):
    external_id: str | None = None
    title: str
    script: str = 'Old North Arabian'
    site: str | None = None
    image_url: str | None = None
    description: str | None = None
    rights: str | None = None
    provenance: str | None = None

class ReadingIn(BaseModel):
    object_id: int
    reading_type: str = 'transliteration'
    transliteration: str | None = None
    arabic_interpretation: str | None = None
    english_interpretation: str | None = None
    reviewer: str | None = None
    confidence: float | None = None
    status: str = 'candidate'

def con():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    c.execute('PRAGMA foreign_keys=ON')
    return c

def bootstrap():
    schema = (ROOT / 'database' / 'schema.sql').read_text(encoding='utf-8')
    views = (ROOT / 'database' / 'views.sql').read_text(encoding='utf-8')
    with con() as c:
        c.executescript(schema)
        c.executescript(views)

@app.on_event('startup')
def startup():
    bootstrap()

@app.get('/api/health')
def health():
    return {'ok': True, 'database': str(DB), 'unicode_range': 'U+10A80-U+10A9F'}

@app.get('/api/objects')
def objects(q: str | None = None, limit: int = 50, offset: int = 0):
    limit = max(1, min(limit, 500)); offset = max(0, offset)
    with con() as c:
        rows = c.execute('SELECT * FROM object_summary WHERE (? IS NULL OR title LIKE ? OR site LIKE ?) ORDER BY id DESC LIMIT ? OFFSET ?',
                         (q, f'%{q}%' if q else None, f'%{q}%' if q else None, limit, offset)).fetchall()
    return [dict(r) for r in rows]

@app.post('/api/objects')
def create_object(item: ObjectIn):
    with con() as c:
        cur = c.execute('INSERT INTO objects(external_id,title,script,site,image_url,description,rights,provenance) VALUES(?,?,?,?,?,?,?,?)', item.model_dump().values())
        return {'id': cur.lastrowid, **item.model_dump()}

@app.get('/api/readings')
def readings(status: str | None = None):
    with con() as c:
        rows = c.execute('SELECT * FROM reading_dashboard WHERE (? IS NULL OR status=?) ORDER BY id DESC', (status, status)).fetchall()
    return [dict(r) for r in rows]

@app.post('/api/readings')
def create_reading(item: ReadingIn):
    if item.confidence is not None and not 0 <= item.confidence <= 1:
        raise HTTPException(422, 'confidence must be between 0 and 1')
    with con() as c:
        cur = c.execute('INSERT INTO readings(object_id,reading_type,transliteration,arabic_interpretation,english_interpretation,reviewer,confidence,status) VALUES(?,?,?,?,?,?,?,?)', tuple(item.model_dump().values()))
        return {'id': cur.lastrowid, **item.model_dump()}

@app.post('/api/import/softr')
async def import_softr(file: UploadFile = File(...)):
    raw = await file.read()
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(413, 'CSV exceeds 10 MiB')
    text = raw.decode('utf-8-sig')
    reader = csv.DictReader(io.StringIO(text))
    count = 0
    with con() as c:
        for row in reader:
            title = row.get('title') or row.get('Title') or row.get('name') or row.get('Name')
            if not title:
                continue
            c.execute('INSERT OR IGNORE INTO objects(external_id,title,script,site,image_url,description,rights,provenance) VALUES(?,?,?,?,?,?,?,?)',
                      (row.get('Record ID') or row.get('record_id'), title, row.get('script') or 'Old North Arabian', row.get('site'), row.get('image') or row.get('image_url'), row.get('description'), row.get('rights'), row.get('provenance')))
            count += 1
    return {'imported': count}

@app.get('/api/export/objects.csv')
def export_objects():
    with con() as c:
        rows = c.execute('SELECT * FROM object_summary ORDER BY id').fetchall()
    out = io.StringIO(); writer = csv.DictWriter(out, fieldnames=rows[0].keys() if rows else ['id','title']); writer.writeheader()
    for r in rows: writer.writerow(dict(r))
    return StreamingResponse(iter([out.getvalue()]), media_type='text/csv', headers={'Content-Disposition':'attachment; filename=thamudic-objects.csv'})
