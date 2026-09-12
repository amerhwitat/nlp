import {readFile} from 'node:fs/promises';
const image=process.argv[2],engine=process.argv[3]??'auto';
if(!image){console.error('usage: ocr_scan image [engine]');process.exit(2)}
const form=new FormData();form.append('file',new Blob([await readFile(image)]),'image');
const r=await fetch(`http://127.0.0.1:8010/api/ocr/scan?engine=${encodeURIComponent(engine)}`,{method:'POST',body:form});
console.log(await r.text());if(!r.ok)process.exit(1);
