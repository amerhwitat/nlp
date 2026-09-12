from __future__ import annotations

import csv
import io
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from pdf_export import export_research_pdf
from pdf_import import import_pdf
from kpi import languages as kpi_languages, summary as kpi_summary

ROOT = Path(__file__).resolve().parents[1]
DB = Path(os.getenv('THAMUDIC_PLATFORM_DB', ROOT / 'data' / 'thamudic_platform.sqlite'))
DB.parent.mkdir(parents=True, exist_ok=True)

app = FastAPI(title='Thamudic Cross-Language Epigraphy API', version='1.1.0')
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
        migration = ROOT / 'database' / 'migrations' / '002_pdf_translation_kpi.sql'
        c.executescript(migration.read_text(encoding='utf-8'))

@app.on_event('startup')
def startup():
    bootstrap()

@app.get('/api/health')
def health():
    return {'ok': True, 'database': str(DB), 'unicode_range': 'U+10A80-U+10A9F', 'api_version': '1.1.0'}

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
        values = item.model_dump()
        cur = c.execute('INSERT INTO objects(external_id,title,script,site,image_url,description,rights,provenance) VALUES(?,?,?,?,?,?,?,?)', tuple(values.values()))
        return {'id': cur.lastrowid, **values}

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
        values = item.model_dump()
        cur = c.execute('INSERT INTO readings(object_id,reading_type,transliteration,arabic_interpretation,english_interpretation,reviewer,confidence,status) VALUES(?,?,?,?,?,?,?,?)', tuple(values.values()))
        return {'id': cur.lastrowid, **values}

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

@app.post('/api/pdf/import')
async def pdf_import(file: UploadFile = File(...), source_id: int | None = None):
    raw = await file.read()
    try:
        result = import_pdf(raw)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    with con() as c:
        cur = c.execute(
            'INSERT INTO pdf_imports(source_id,filename,sha256,page_count,status,extracted_chars,warnings_json) VALUES(?,?,?,?,?,?,?)',
            (source_id, result.filename, result.sha256, result.page_count, 'completed', result.total_chars, json.dumps(result.warnings, ensure_ascii=False)),
        )
        import_id = cur.lastrowid
    return {'id': import_id, 'filename': result.filename, 'sha256': result.sha256, 'page_count': result.page_count,
            'total_chars': result.total_chars, 'warnings': result.warnings,
            'pages': [{'page_number': p.page_number, 'text': p.text, 'char_count': p.char_count} for p in result.pages]}

@app.post('/api/pdf/export')
def pdf_export(report: dict[str, Any]):
    pdf, manifest = export_research_pdf(report)
    with con() as c:
        c.execute('INSERT INTO pdf_exports(report_id,object_id,filename,sha256,report_type,manifest_json) VALUES(?,?,?,?,?,?)',
                  (manifest['report_id'], report.get('object_id'), f"{manifest['report_id']}.pdf", manifest['pdf_sha256'], report.get('report_type', 'research'), json.dumps(manifest, ensure_ascii=False)))
    return Response(pdf, media_type='application/pdf', headers={'Content-Disposition': f"attachment; filename={manifest['report_id']}.pdf", 'X-Report-ID': manifest['report_id']})

@app.get('/api/kpis/summary')
def kpis_summary():
    with con() as c:
        return kpi_summary(c)

@app.get('/api/kpis/languages')
def kpis_languages():
    with con() as c:
        return kpi_languages(c)
