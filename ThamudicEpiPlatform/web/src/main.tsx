import React, {useEffect, useState} from 'react';
import {createRoot} from 'react-dom/client';
import './styles.css';

const API=import.meta.env.VITE_API_URL||'http://127.0.0.1:8010/api';
function App(){
 const [objects,setObjects]=useState<any[]>([]); const [q,setQ]=useState(''); const [status,setStatus]=useState('');
 async function load(){setStatus('Loading…'); const r=await fetch(`${API}/objects?q=${encodeURIComponent(q)}`); setObjects(await r.json()); setStatus('Ready');}
 useEffect(()=>{load()},[]);
 async function importCsv(e:any){const f=e.target.files?.[0]; if(!f)return; const fd=new FormData(); fd.append('file',f); const r=await fetch(`${API}/import/softr`,{method:'POST',body:fd}); setStatus(JSON.stringify(await r.json())); load();}
 return <main><header><div><h1>𐪀 Thamudic Epigraphy</h1><p>Research corpus · provenance · readings · Unicode</p></div><a href={`${API}/export/objects.csv`}>Export CSV</a></header>
 <section className="toolbar"><input value={q} onChange={e=>setQ(e.target.value)} placeholder="Search objects, sites…"/><button onClick={load}>Search</button><label className="upload">Import Softr CSV<input type="file" accept=".csv" onChange={importCsv}/></label></section>
 <section className="stats"><b>{objects.length}</b><span>visible objects</span><b>U+10A80–10A9F</b><span>Old North Arabian</span></section>
 <section className="grid">{objects.map(o=><article key={o.id}><div className="glyph">𐪀𐪁𐪂</div><h2>{o.title}</h2><p>{o.script} · {o.site||'Unknown site'}</p><p>{o.annotation_count||0} annotations · {o.reading_count||0} readings</p><small>{o.rights||'Rights not recorded'}</small></article>)}</section>
 <footer>{status}</footer></main>
}
createRoot(document.getElementById('root')!).render(<App/>);
