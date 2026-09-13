#!/usr/bin/env python3
"""Resilient compatibility launcher for the unified Thamudic/NLP scanners.

Fixes the historical `Native worker timed out after 180s` failure by routing
image OCR through the bounded, process-cleaning worker in
`python/thamudic/resilient_ocr.py`. OCR failure is recoverable and does not
terminate the GUI/NLP pipeline.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"python"))

from thamudic.resilient_ocr import DEFAULT_TIMEOUT, ocr_image


def _patch(module):
    original_extract=module.extract
    worker=ROOT/"python"/"thamudic"/"ocr_worker.py"
    def extract(path):
        p=Path(path).expanduser(); suffix=p.suffix.casefold()
        if suffix in {".png",".jpg",".jpeg",".webp",".bmp",".tif",".tiff"}:
            r=ocr_image(p,worker,timeout=DEFAULT_TIMEOUT)
            meta={"provider":"easyocr-subprocess","media_type":"image","source_file":str(p),"ocr_available":bool(r.get("ok") and r.get("text")),"ocr_error":"" if r.get("ok") else f"{r.get('error_type','Error')}: {r.get('error','OCR unavailable')}","ocr_timed_out":bool(r.get("timed_out")),"ocr_recoverable":True}
            for k in ("detections","ocr_confidence"):
                if k in r: meta[k]=r[k]
            return str(r.get("text","")),meta
        return original_extract(path)
    module.extract=extract
    return module


def main(argv=None):
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode",choices=("nlp","thamudic"),default="nlp")
    known,rest=parser.parse_known_args(argv)
    module_name="NLPScanner_AllInOne" if known.mode=="nlp" else "ThamudicScanner_AllInOne"
    module=__import__(module_name,fromlist=["*"])
    module=_patch(module)
    return module.cli(rest,module_name)

if __name__=="__main__":
    raise SystemExit(main())
