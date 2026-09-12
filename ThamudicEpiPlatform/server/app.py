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

from server.pdf_export import export_research_pdf
from server.pdf_import import import_pdf
from server.kpi import languages as kpi_languages, summary as kpi_summary
from server.ocr_scanner import scan_image
from server.ocr_geometry import analyze_geometry, build_engine_options
from server.translation_engine import prepare_translation, proof_translation
from server.chatbot import ChatMessage, ResearchChatbot, health as chatbot_health

ROOT = Path(__file__).resolve().parents[1]
DB = Path(os.getenv('THAMUDIC_PLATFORM_DB', ROOT / 'data' / 'thamudic_platform.sqlite'))
DB.parent.mkdir(parents=True, exist_ok=True)
app = FastAPI(title='Thamudic Cross-Language Epigraphy API', version='1.4.0')
origins = [x.strip() for x in os.getenv('CORS_ORIGINS', 'http://localhost:5173,http://127.0.0.1:5173').split(',') if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=['*'], allow_headers=['*'])

class ObjectIn(BaseModel):
    external_id: str | None = None; title: str; script: str = 'Old North Arabian'; site: str | None = None; image_url: str | None = None; description: str | None = None; rights: str | None = None; provenance: str | None = None
class ReadingIn(BaseModel):
    object_id: int; reading_type: str = 'transliteration'; transliteration: str | None = None; arabic_interpretation: str | None = None; english_interpretation: str | None = None; reviewer: str | None = None; confidence: float | None = None; status: str = 'candidate'
class GeometryIn(BaseModel):
    width: int; height: int; edge_density: float = 0.0; contrast: float = 0.0; blur: float = 0.0; orientation_hint: str | None = None
class TranslationIn(BaseModel):
    source_text: str; source_language: str; target_language: str; mode: str = 'meaning'; output: str = ''; confidence: float = 0.0; proof_notes: list[str] = []; alternatives: list[str] = []; provenance: dict[str, Any] = {}
class ChatIn(BaseModel):
    messages: list[ChatMessage]; context: list[dict[str, Any]] = []

def con():
    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row; c.execute('PRAGMA foreign_keys=ON'); return c

def bootstrap():
    with con() as c:
        c.executescript((ROOT / 'database/schema.sql').read_text(encoding='utf-8'))
        c.executescript((ROOT / 'database/views.sql').read_text(encoding='utf-8'))
        for name in ('002_pdf_translation_kpi.sql','003_ocr_jobs.sql','004_orientation_translation_chat.sql'):
            c.executescript((ROOT / 'database/migrations' / name).read_text(encoding='utf-8'))

@app.on_event('startup')
def startup(): bootstrap()

@app.get('/api/health')
def health():
    return {'ok': True, 'database': str(DB), 'api_version': '1.4.0', 'ocr': ['auto','kraken','tesseract','quality-only'], 'geometry': ['rtl','ltr','ttb','btt','spiral','reverse','skewed','weathered'], 'translation': ['literal','meaning','interlinear','scholarly'], 'source_target_languages': ['ancient','zh','ja','modern-bcp47'], 'chatbot': chatbot_health()}

@app.get('/api/objects')
def objects(q: str | None = None, limit: int = 50, offset: int = 0):
    limit = max(1, min(limit, 500)); offset = max(0, offset)
    with con() as c: rows = c.execute('SELECT * FROM object_summary WHERE (? IS NULL OR title LIKE ? OR site LIKE ?) ORDER BY id DESC LIMIT ? OFFSET ?', (q, f'%{q}%' if q else None, f'%{q}%' if q else None, limit, offset)).fetchall()
    return [dict(r) for r in rows]

@app.post('/api/objects')
def create_object(item: ObjectIn):
    with con() as c:
        values=item.model_dump(); cur=c.execute('INSERT INTO objects(external_id,title,script,site,image_url,description,rights,provenance) VALUES(?,?,?,?,?,?,?,?)', tuple(values.values())); return {'id':cur.lastrowid,**values}

@app.get('/api/readings')
def readings(status: str | None = None):
    with con() as c: rows=c.execute('SELECT * FROM reading_dashboard WHERE (? IS NULL OR status=?) ORDER BY id DESC',(status,status)).fetchall()
    return [dict(r) for r in rows]

@app.post('/api/readings')
def create_reading(item: ReadingIn):
    if item.confidence is not None and not 0 <= item.confidence <= 1: raise HTTPException(422,'confidence must be between 0 and 1')
    with con() as c:
        values=item.model_dump(); cur=c.execute('INSERT INTO readings(object_id,reading_type,transliteration,arabic_interpretation,english_interpretation,reviewer,confidence,status) VALUES(?,?,?,?,?,?,?,?)',tuple(values.values())); return {'id':cur.lastrowid,**values}

@app.post('/api/ocr/geometry')
def ocr_geometry(item: GeometryIn):
    h=analyze_geometry(item.width,item.height,edge_density=item.edge_density,contrast=item.contrast,blur=item.blur,orientation_hint=item.orientation_hint); return {'hypothesis':h.__dict__,'engine_options':build_engine_options(h)}

@app.post('/api/translation/proof')
def translation_proof(item: TranslationIn):
    return proof_translation(prepare_translation(item.source_text,item.source_language,item.target_language,item.mode,output=item.output,confidence=item.confidence,proof_notes=item.proof_notes,alternatives=item.alternatives,provenance=item.provenance))

@app.post('/api/chat')
def chat(item: ChatIn):
    return ResearchChatbot().generate(item.messages,item.context).__dict__

@app.post('/api/import/softr')
async def import_softr(file: UploadFile=File(...)):
    raw=await file.read()
    if len(raw)>10*1024*1024: raise HTTPException(413,'CSV exceeds 10 MiB')
    reader=csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))); count=0
    with con() as c:
        for row in reader:
            title=row.get('title') or row.get('Title') or row.get('name') or row.get('Name')
            if not title: continue
            c.execute('INSERT OR IGNORE INTO objects(external_id,title,script,site,image_url,description,rights,provenance) VALUES(?,?,?,?,?,?,?,?)',(row.get('Record ID') or row.get('record_id'),title,row.get('script') or 'Old North Arabian',row.get('site'),row.get('image') or row.get('image_url'),row.get('description'),row.get('rights'),row.get('provenance'))); count+=1
    return {'imported':count}

@app.get('/api/export/objects.csv')
def export_objects():
    with con() as c: rows=c.execute('SELECT * FROM object_summary ORDER BY id').fetchall()
    out=io.StringIO(); writer=csv.DictWriter(out,fieldnames=rows[0].keys() if rows else ['id','title']); writer.writeheader()
    for r in rows: writer.writerow(dict(r))
    return StreamingResponse(iter([out.getvalue()]),media_type='text/csv',headers={'Content-Disposition':'attachment; filename=thamudic-objects.csv'})

@app.post('/api/ocr/scan')
async def ocr_scan(file: UploadFile=File(...),engine: str='auto',tesseract_lang: str|None=None,kraken_model: str|None=None):
    raw=await file.read()
    try: result=scan_image(raw,engine=engine,tesseract_lang=tesseract_lang,kraken_model=kraken_model)
    except ValueError as exc: raise HTTPException(422,str(exc)) from exc
    with con() as c: c.execute('INSERT INTO ocr_jobs(filename,source_sha256,engine,status,confidence,script_candidates_json,warnings_json,preprocessing_json) VALUES(?,?,?,?,?,?,?,?)',(file.filename,result.source_sha256,result.engine,'completed' if result.text else 'error',result.confidence,json.dumps(result.script_candidates,ensure_ascii=False),json.dumps(result.warnings,ensure_ascii=False),json.dumps(result.preprocessing,ensure_ascii=False)))
    return {'filename':file.filename,'content_type':file.content_type,**result.to_dict(),'provenance':{'source_filename':file.filename,'sha256':result.source_sha256,'recognition_only':True,'translation_required_review':True}}

@app.post('/api/pdf/import')
async def pdf_import(file: UploadFile=File(...),source_id: int|None=None):
    raw=await file.read()
    try: result=import_pdf(raw)
    except ValueError as exc: raise HTTPException(422,str(exc)) from exc
    with con() as c: cur=c.execute('INSERT INTO pdf_imports(source_id,filename,sha256,page_count,status,extracted_chars,warnings_json) VALUES(?,?,?,?,?,?,?)',(source_id,result.filename,result.sha256,result.page_count,'completed',result.total_chars,json.dumps(result.warnings,ensure_ascii=False)))
    return {'id':cur.lastrowid,'filename':result.filename,'sha256':result.sha256,'page_count':result.page_count,'total_chars':result.total_chars,'warnings':result.warnings,'pages':[{'page_number':p.page_number,'text':p.text,'char_count':p.char_count} for p in result.pages]}

@app.post('/api/pdf/export')
def pdf_export(report: dict[str,Any]):
    pdf,manifest=export_research_pdf(report)
    with con() as c: c.execute('INSERT INTO pdf_exports(report_id,object_id,filename,sha256,report_type,manifest_json) VALUES(?,?,?,?,?,?)',(manifest['report_id'],report.get('object_id'),f"{manifest['report_id']}.pdf",manifest['pdf_sha256'],report.get('report_type','research'),json.dumps(manifest,ensure_ascii=False)))
    return Response(pdf,media_type='application/pdf',headers={'Content-Disposition':f"attachment; filename={manifest['report_id']}.pdf",'X-Report-ID':manifest['report_id']})

@app.get('/api/kpis/summary')
def kpis_summary():
    with con() as c: return kpi_summary(c)
@app.get('/api/kpis/languages')
def kpis_languages():
    with con() as c: return kpi_languages(c)
