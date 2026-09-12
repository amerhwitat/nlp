import React,{useEffect,useState} from 'react';
import {createRoot} from 'react-dom/client';
import './styles.css';

const API=import.meta.env.VITE_API_URL||'http://127.0.0.1:8010/api';
type KPI=Record<string,number|string>;
function App(){
 const [objects,setObjects]=useState<any[]>([]),[q,setQ]=useState(''),[status,setStatus]=useState(''),[kpi,setKpi]=useState<KPI>({}),[ocr,setOcr]=useState<any|null>(null);
 async function load(){setStatus('Loading…');const [o,k]=await Promise.all([fetch(`${API}/objects?q=${encodeURIComponent(q)}`),fetch(`${API}/kpis/summary`)]);setObjects(await o.json());setKpi(await k.json());setStatus('Ready');}
 useEffect(()=>{load()},[]);
 async function importCsv(e:any){const f=e.target.files?.[0];if(!f)return;const fd=new FormData();fd.append('file',f);const r=await fetch(`${API}/import/softr`,{method:'POST',body:fd});setStatus(JSON.stringify(await r.json()));load();}
 async function scan(e:any){const f=e.target.files?.[0];if(!f)return;setStatus('Scanning image…');const fd=new FormData();fd.append('file',f);const r=await fetch(`${API}/ocr/scan?engine=auto`,{method:'POST',body:fd});const data=await r.json();setOcr(data);setStatus(r.ok?'OCR scan complete':'OCR scan failed');}
 return <main><header><div><h1>𐪀 Thamudic Epigraphy + Intelligent OCR</h1><p>Historical objects · provenance · Unicode · OCR · transliteration review</p></div><a href={`${API}/export/objects.csv`}>Export CSV</a></header>
 <section className="toolbar"><input value={q} onChange={e=>setQ(e.target.value)} placeholder="Search objects, sites…"/><button onClick={load}>Search</button><label className="upload">Import Softr CSV<input type="file" accept=".csv" onChange={importCsv}/></label><label className="upload">Scan image<input type="file" accept="image/*" onChange={scan}/></label></section>
 <section className="kpi"><h2>Application KPIs</h2><div className="kpiGrid">{Object.entries(kpi).map(([key,value])=><div className="kpiCard" key={key}><b>{String(value)}</b><span>{key.replaceAll('_',' ')}</span></div>)}</div></section>
 {ocr&&<section className="ocrPanel"><h2>Intelligent OCR result</h2><p><b>Engine:</b> {ocr.engine} · <b>Confidence:</b> {ocr.confidence}</p><p><b>Scripts:</b> {(ocr.script_candidates||[]).map((x:any)=>`${x.script} ${x.score}`).join(' · ')||'No script identified from recognized text'}</p><textarea readOnly value={ocr.text||''} placeholder="Recognition output"/><p className="warning">{(ocr.warnings||[]).join(' · ')}</p><small>SHA-256: {ocr.source_sha256} · Recognition is not translation; scholarly review is required.</small></section>}
 <section className="stats"><b>{objects.length}</b><span>visible objects</span><b>U+10A80–10A9F</b><span>Old North Arabian</span></section>
 <section className="grid">{objects.map(o=><article key={o.id}><div className="glyph">𐪀𐪁𐪂</div><h2>{o.title}</h2><p>{o.script} · {o.site||'Unknown site'}</p><p>{o.annotation_count||0} annotations · {o.reading_count||0} readings</p><small>{o.rights||'Rights not recorded'}</small></article>)}</section>
 <footer>{status}</footer></main>
}
createRoot(document.getElementById('root')!).render(<App/>);
