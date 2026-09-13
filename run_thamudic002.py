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
            if r.get("ok"):
                return str(r.get("text","")), {"provider":"easyocr-subprocess","media_type":"image","source_file":str(p),"ocr_available":bool(r.get("text")),"ocr_error":"","ocr_timed_out":False,"ocr_recoverable":True,"detections":r.get("detections",0),"ocr_confidence":r.get("ocr_confidence",0.0)}
            return "", {"provider":"easyocr-subprocess","media_type":"image","source_file":str(p),"ocr_available":False,"ocr_error":f"{r.get('error_type','Error')}: {r.get('error','OCR unavailable')}","ocr_timed_out":bool(r.get("timed_out")),"ocr_recoverable":True}
        return original_extract(path)

    module.extract=extract
    return module


def main(argv=None):
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode",choices=("nlp","thamudic"),default="nlp")
    known,rest=parser.parse_known_args(argv)
    module_name="NLPScanner_AllInOne" if known.mode=="nlp" else "ThamudicScanner_AllInOne"
    module=__import__(f"python.{module_name}" if (ROOT/"python"/f"{module_name}.py").exists() else module_name,fromlist=["*"])
    # The python directory is also placed directly on sys.path for installations
    # where it is not a package.
    module=_patch(module)
    return module.cli(rest,module_name)

if __name__=="__main__":
    raise SystemExit(main())
