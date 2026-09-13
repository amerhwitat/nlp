import express from 'express';
import cors from 'cors';
import multer from 'multer';
import Database from 'better-sqlite3';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.dirname(fileURLToPath(import.meta.url));
const dataDir = process.env.ARTIFACT_DATA_DIR || path.join(root, 'data');
fs.mkdirSync(dataDir, { recursive: true });
const db = new Database(path.join(dataDir, 'artifacts.sqlite'));
db.pragma('journal_mode = WAL');
db.pragma('foreign_keys = ON');
db.exec(`
CREATE TABLE IF NOT EXISTS artifacts (
 id TEXT PRIMARY KEY, title TEXT DEFAULT '', source TEXT DEFAULT '', media_type TEXT DEFAULT 'text',
 original_text TEXT DEFAULT '', source_language TEXT DEFAULT '', script_variant TEXT DEFAULT '', target_language TEXT DEFAULT '',
 transliteration TEXT, translation TEXT, translation_status TEXT, confidence TEXT, provider TEXT, provenance TEXT,
 object_type TEXT, culture TEXT, period_key TEXT, period_name TEXT, date_start TEXT, date_end TEXT,
 site TEXT, region TEXT, country TEXT, current_location TEXT, material TEXT, technique TEXT, description TEXT,
 creator TEXT, latitude REAL, longitude REAL, reviewer TEXT, metadata_json TEXT DEFAULT '{}', tags_json TEXT DEFAULT '[]',
 created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_artifacts_search ON artifacts(source_language,script_variant,period_key,object_type,country);
CREATE TABLE IF NOT EXISTS artifact_scans (id TEXT PRIMARY KEY, artifact_id TEXT NOT NULL, result_json TEXT NOT NULL, created_at TEXT NOT NULL, FOREIGN KEY(artifact_id) REFERENCES artifacts(id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS artifact_media (id TEXT PRIMARY KEY, artifact_id TEXT NOT NULL, media_path TEXT, media_type TEXT, sha256 TEXT, metadata_json TEXT DEFAULT '{}', created_at TEXT NOT NULL, FOREIGN KEY(artifact_id) REFERENCES artifacts(id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS artifact_annotations (id TEXT PRIMARY KEY, artifact_id TEXT NOT NULL, annotation_json TEXT NOT NULL, created_at TEXT NOT NULL, FOREIGN KEY(artifact_id) REFERENCES artifacts(id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS artifact_provenance (id TEXT PRIMARY KEY, artifact_id TEXT NOT NULL, source_name TEXT, source_url TEXT, rights TEXT, license TEXT, record_id TEXT, notes TEXT, created_at TEXT NOT NULL, FOREIGN KEY(artifact_id) REFERENCES artifacts(id) ON DELETE CASCADE);
`);
const now = () => new Date().toISOString();
const id = () => crypto.randomUUID().replaceAll('-', '');
const json = (v, fallback) => JSON.stringify(v ?? fallback);
const decode = row => row ? {...row, metadata: JSON.parse(row.metadata_json || '{}'), tags: JSON.parse(row.tags_json || '[]')} : null;

const ONA_FIRST = 0x10a80, ONA_LAST = 0x10a9f;
const translit = new Map([
  ['𐪀','ʾ'],['𐪁','b'],['𐪂','g'],['𐪃','d'],['𐪄','h'],['𐪅','w'],['𐪆','z'],['𐪇','ḥ'],['𐪈','ṭ'],['𐪉','y'],['𐪊','k'],['𐪋','l'],['𐪌','m'],['𐪍','n'],['𐪎','s'],['𐪏','ʿ'],['𐪐','p'],['𐪑','ṣ'],['𐪒','q'],['𐪓','r'],['𐪔','š'],['𐪕','t'],['𐪖','ṯ'],['𐪗','f'],['𐪘','ḏ'],['𐪙','ḍ'],['𐪚','ġ'],['𐪛','ḫ'],['𐪜','ẓ']
]);
function scanText(text, script='Ancient North Arabian') {
  const chars = [...String(text)];
  const matches = chars.filter(ch => { const cp = ch.codePointAt(0); return cp >= ONA_FIRST && cp <= ONA_LAST; });
  return {matched: matches.length > 0, script_variant: script, language: 'Ancient North Arabian', count: matches.length,
    codepoints: matches.map(ch => `U+${ch.codePointAt(0).toString(16).toUpperCase().padStart(4,'0')}`),
    transliteration: chars.map(ch => translit.get(ch) ?? ch).join(''), recognition_status: matches.length ? 'candidate-glyphs-detected' : 'no-target-glyphs-detected'};
}

const app = express();
app.use(cors()); app.use(express.json({limit:'10mb'}));
const upload = multer({dest: path.join(dataDir, 'uploads')});
app.get('/health', (_,res) => res.json({ok:true, service:'ancient-artifacts-node', database:'sqlite', evidence_policy:'evidence-first'}));
app.get('/api/artifacts/stats', (_,res) => {
  const tables=['artifacts','artifact_scans','artifact_media','artifact_annotations','artifact_provenance'];
  const counts=Object.fromEntries(tables.map(t=>[t,db.prepare(`SELECT COUNT(*) n FROM ${t}`).get().n]));
  res.json({schema_version:'2',...counts});
});
app.get('/api/artifacts', (req,res) => {
  const q=String(req.query.q||''), filters=[]; const args=[];
  if(q){filters.push('(title LIKE @q OR original_text LIKE @q OR transliteration LIKE @q OR translation LIKE @q OR description LIKE @q OR tags_json LIKE @q OR source LIKE @q)');args.push({q:`%${q}%`});}
  for(const [field,key] of [['source_language','source_language'],['script_variant','script_variant'],['period_key','period_key'],['object_type','object_type'],['country','country']]) if(req.query[key]){filters.push(`${field}=@${field}`);args.push({[field]:String(req.query[key])});}
  const limit=Math.min(Math.max(Number(req.query.limit||100),1),1000);
  const sql=`SELECT * FROM artifacts ${filters.length?'WHERE '+filters.join(' AND '):''} ORDER BY updated_at DESC LIMIT ${limit}`;
  res.json({artifacts:db.prepare(sql).all(Object.assign({},...args)).map(decode)});
});
app.get('/api/artifacts/:id', (req,res) => {
  const row=decode(db.prepare('SELECT * FROM artifacts WHERE id=?').get(req.params.id));
  if(!row) return res.status(404).json({error:'artifact not found'});
  row.scans=db.prepare('SELECT * FROM artifact_scans WHERE artifact_id=? ORDER BY created_at').all(req.params.id).map(x=>JSON.parse(x.result_json));
  row.media=db.prepare('SELECT * FROM artifact_media WHERE artifact_id=? ORDER BY created_at').all(req.params.id);
  row.annotations=db.prepare('SELECT * FROM artifact_annotations WHERE artifact_id=? ORDER BY created_at').all(req.params.id).map(x=>JSON.parse(x.annotation_json));
  row.provenance=db.prepare('SELECT * FROM artifact_provenance WHERE artifact_id=? ORDER BY created_at').all(req.params.id);
  res.json(row);
});
app.post('/api/scan', (req,res) => res.json(scanText(req.body?.text||'', req.body?.script||'Ancient North Arabian')));
app.post('/api/artifacts', (req,res) => {
  const r=req.body||{}, artifactId=r.id||id(), timestamp=now();
  db.prepare(`INSERT OR REPLACE INTO artifacts (id,title,source,media_type,original_text,source_language,script_variant,target_language,transliteration,translation,translation_status,confidence,provider,provenance,object_type,culture,period_key,period_name,date_start,date_end,site,region,country,current_location,material,technique,description,creator,latitude,longitude,reviewer,metadata_json,tags_json,created_at,updated_at) VALUES (@id,@title,@source,@media_type,@original_text,@source_language,@script_variant,@target_language,@transliteration,@translation,@translation_status,@confidence,@provider,@provenance,@object_type,@culture,@period_key,@period_name,@date_start,@date_end,@site,@region,@country,@current_location,@material,@technique,@description,@creator,@latitude,@longitude,@reviewer,@metadata_json,@tags_json,@created_at,@updated_at)`).run({id:artifactId,title:r.title||'',source:r.source||'',media_type:r.media_type||'text',original_text:r.original_text||r.text||'',source_language:r.source_language||r.language||'',script_variant:r.script_variant||r.script||'',target_language:r.target_language||'',transliteration:r.transliteration||null,translation:r.translation||null,translation_status:r.translation_status||null,confidence:String(r.confidence??''),provider:r.provider||null,provenance:r.provenance||null,object_type:r.object_type||null,culture:r.culture||null,period_key:r.period_key||null,period_name:r.period_name||null,date_start:r.date_start||null,date_end:r.date_end||null,site:r.site||null,region:r.region||null,country:r.country||null,current_location:r.current_location||null,material:r.material||null,technique:r.technique||null,description:r.description||null,creator:r.creator||null,latitude:r.latitude??null,longitude:r.longitude??null,reviewer:r.reviewer||null,metadata_json:json(r.metadata,{}),tags_json:json(r.tags,[]),created_at:timestamp,updated_at:timestamp});
  res.status(201).json({id:artifactId,artifact:decode(db.prepare('SELECT * FROM artifacts WHERE id=?').get(artifactId))});
});
app.post('/api/artifacts/:id/annotations',(req,res)=>{if(!db.prepare('SELECT id FROM artifacts WHERE id=?').get(req.params.id))return res.status(404).json({error:'artifact not found'});const aid=id();db.prepare('INSERT INTO artifact_annotations VALUES(?,?,?,?)').run(aid,req.params.id,json(req.body,{}),now());res.status(201).json({id:aid});});
app.post('/api/artifacts/:id/provenance',(req,res)=>{if(!db.prepare('SELECT id FROM artifacts WHERE id=?').get(req.params.id))return res.status(404).json({error:'artifact not found'});const p=req.body||{},pid=id();db.prepare('INSERT INTO artifact_provenance VALUES(?,?,?,?,?,?,?,?,?)').run(pid,req.params.id,p.source_name||null,p.source_url||null,p.rights||null,p.license||null,p.record_id||null,p.notes||null,now());res.status(201).json({id:pid});});
app.post('/api/artifacts/:id/media',upload.single('file'),(req,res)=>{if(!req.file)return res.status(400).json({error:'file required'});if(!db.prepare('SELECT id FROM artifacts WHERE id=?').get(req.params.id))return res.status(404).json({error:'artifact not found'});const bytes=fs.readFileSync(req.file.path),mid=id();db.prepare('INSERT INTO artifact_media VALUES(?,?,?,?,?,?,?)').run(mid,req.params.id,req.file.path,req.file.mimetype,crypto.createHash('sha256').update(bytes).digest('hex'),json(req.body,{}),now());res.status(201).json({id:mid,sha256:crypto.createHash('sha256').update(bytes).digest('hex')});});
app.use(express.static(path.join(root,'web')));
const port=Number(process.env.PORT||8090); app.listen(port,()=>console.log(`Ancient Artifact Node API listening on http://127.0.0.1:${port}`));
