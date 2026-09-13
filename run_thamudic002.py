#!/usr/bin/env python3
"""Resilient Thamudic/NLP all-in-one launcher.

EasyOCR runs in an isolated native worker with a bounded timeout. A timeout is
recorded as recoverable metadata and glyph segmentation/NLP processing continues.
"""
from __future__ import annotations
import argparse,sys
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
    cls=app.NLPApp if mode=="nlp" else app.App
    cls(db_path=db).mainloop()

def main(argv=None):
    p=argparse.ArgumentParser(description="Resilient Thamudic/NLP scanner")
    p.add_argument("--mode",choices=("nlp","thamudic"),default="nlp")
    p.add_argument("--db",default="ancient_objects.sqlite")
    a=p.parse_args(argv); launch(a.mode,a.db); return 0
if __name__=="__main__":raise SystemExit(main())
