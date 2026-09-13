"""Isolated OCR worker for native image/PDF dependencies.

EasyOCR/PyTorch can load native DLLs that conflict with the parent Tkinter process.
Running OCR in a short-lived child process keeps a native-library crash from taking
down the desktop GUI. The parent communicates through JSON on stdout.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _ocr_image(path: Path, languages: list[str]) -> dict:
    import easyocr  # type: ignore

    reader = easyocr.Reader(languages, gpu=False, verbose=False)
    rows = reader.readtext(str(path), detail=1, paragraph=False)
    rows = sorted(rows, key=lambda x: (min(p[1] for p in x[0]), min(p[0] for p in x[0])))
    text = "\n".join(str(x[1]) for x in rows).strip()
    confidence = sum(float(x[2]) for x in rows) / len(rows) if rows else 0.0
    return {
        "text": text,
        "provider": "easyocr-subprocess",
        "detections": len(rows),
        "ocr_confidence": round(confidence, 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Isolated EasyOCR worker")
    parser.add_argument("image", type=Path)
    parser.add_argument("--languages", default="en,ar")
    args = parser.parse_args()
    try:
        if not args.image.exists():
            raise FileNotFoundError(args.image)
        languages = [x.strip() for x in args.languages.split(",") if x.strip()] or ["en", "ar"]
        result = _ocr_image(args.image, languages)
        print(json.dumps({"ok": True, **result}, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
