#!/usr/bin/env python3
"""Resilient Thamudic/NLP all-in-one launcher."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"python"))
from thamudic.resilient_ocr import DEFAULT_TIMEOUT,ocr_image

def launch(mode="nlp",db="ancient_objects.sqlite"):
    import thamudic_all_in_one as app
    original=app.scan_image_safe
    worker=ROOT/"python"/"thamudic"/"ocr_worker.py"
    def safe(path,**kwargs):
        r=ocr_image(path,worker,timeout=DEFAULT_TIMEOUT)
        hint=str(r.get("text","")) if r.get("ok") else kwargs.get("text_hint","")
        payload=original(path,**kwargs,text_hint=hint)
        if payload.get("ok"):
            payload["result"]["ocr"]={"provider":r.get("provider","easyocr-subprocess"),"ocr_available":bool(r.get("ok") and r.get("text")),"ocr_error":"" if r.get("ok") else f"{r.get('error_type','Error')}: {r.get('error','OCR unavailable')}","ocr_timed_out":bool(r.get("timed_out")),"ocr_recoverable":True,"ocr_confidence":r.get("ocr_confidence",0.0),"detections":r.get("detections",0),"source_file":str(path)}
            payload["result"]["recognition_status"]="easyocr_text_plus_glyph_segmentation" if r.get("ok") and r.get("text") else "glyph_segmentation_after_recoverable_ocr_failure"
        return payload
    app.scan_image_safe=safe
    (app.NLPApp if mode=="nlp" else app.App)(db_path=db).mainloop()

def cli(argv=None):
    p=argparse.ArgumentParser(description="Resilient Thamudic/NLP scanner")
    p.add_argument("--mode",choices=("nlp","thamudic"),default="nlp")
    p.add_argument("--db",default="ancient_objects.sqlite")
    p.add_argument("--file")
    p.add_argument("--text",default="")
    p.add_argument("--scan",action="store_true")
    p.add_argument("--translate",action="store_true")
    p.add_argument("--script",default="Old North Arabian / Thamudic")
    p.add_argument("--target",default="en",choices=("en","ar"))
    a=p.parse_args(argv)
    import thamudic_all_in_one as app
    if a.text and a.scan:
        print(json.dumps(app.scan(a.text,a.script),ensure_ascii=False,indent=2)); return 0
    if a.text and a.translate:
        print(json.dumps(app.build_text_outputs(a.text),ensure_ascii=False,indent=2)); return 0
    if a.file:
        worker=ROOT/"python"/"thamudic"/"ocr_worker.py"; r=ocr_image(a.file,worker,timeout=DEFAULT_TIMEOUT)
        if r.get("ok"):
            result=app.scan_image(a.file,text_hint=str(r.get("text","")))
        else:
            result=app.scan_image(a.file,text_hint="")
        result["ocr"]={"provider":r.get("provider","easyocr-subprocess"),"ocr_available":bool(r.get("ok") and r.get("text")),"ocr_error":"" if r.get("ok") else f"{r.get('error_type','Error')}: {r.get('error','OCR unavailable')}","ocr_timed_out":bool(r.get("timed_out")),"ocr_recoverable":True}
        print(json.dumps(result,ensure_ascii=False,indent=2)); return 0
    launch(a.mode,a.db); return 0

def main(argv=None): return cli(argv)
if __name__=="__main__":raise SystemExit(main())
