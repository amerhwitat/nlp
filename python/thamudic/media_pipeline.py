"""Crash-safe image/PDF media extraction for the desktop scanner.

Text PDFs are handled in-process with pypdf. Native EasyOCR/PyTorch and pypdfium2
work is isolated in ``ocr_worker.py`` so Windows native DLL/OpenMP failures cannot
terminate the Tkinter parent process. OCR failures are returned as actionable
metadata instead of crashing or fabricating a reading.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

from .ancient_translation import translate
from .source_language_scanner import scan_source_language

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
_TEXT_SUFFIXES = {".txt", ".md", ".csv"}


def _pdf_text(path: Path) -> tuple[str, dict[str, Any]]:
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PDF text extraction requires pypdf. Install python/requirements.txt dependencies.") from exc
    try:
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages).strip()
        return text, {"provider": "pypdf", "pages": len(reader.pages), "scanned_pages": not bool(text)}
    except Exception as exc:
        raise RuntimeError(f"Could not read PDF '{path.name}': {type(exc).__name__}: {exc}") from exc


def _worker_path() -> Path:
    return Path(__file__).with_name("ocr_worker.py")


def _worker_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    return env


def _easyocr_image(path: Path, languages: list[str] | None = None) -> tuple[str, dict[str, Any]]:
    """Run EasyOCR out-of-process so native DLL failures cannot kill the GUI."""
    worker = _worker_path()
    if not worker.exists():
        raise RuntimeError(f"OCR worker is missing: {worker}")
    langs = languages or ["en", "ar"]
    timeout = max(30, int(os.environ.get("THAMUDIC_OCR_TIMEOUT", "180")))
    command = [sys.executable, str(worker), str(path), "--languages", ",".join(langs)]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, env=_worker_env(), check=False)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"OCR timed out after {timeout}s; the GUI was protected and remains running.") from exc
    except OSError as exc:
        raise RuntimeError(f"Could not start isolated OCR worker: {exc}") from exc
    stdout = (completed.stdout or "").strip()
    if not stdout:
        detail = (completed.stderr or "").strip()
        raise RuntimeError("OCR worker exited without a result" + (f": {detail[-1000:]}" if detail else f" (exit code {completed.returncode})"))
    try:
        payload = json.loads(stdout.splitlines()[-1])
    except json.JSONDecodeError as exc:
        detail = (completed.stderr or "").strip()
        raise RuntimeError(f"OCR worker returned invalid JSON{': ' + detail[-1000:] if detail else ''}") from exc
    if not payload.get("ok"):
        message = payload.get("error") or f"worker exit code {completed.returncode}"
        raise RuntimeError(f"OCR unavailable: {payload.get('error_type', 'Error')}: {message}")
    return str(payload.get("text", "")), {
        "provider": payload.get("provider", "easyocr-subprocess"),
        "detections": int(payload.get("detections", 0)),
        "ocr_confidence": float(payload.get("ocr_confidence", 0.0)),
    }


def _render_pdf_page(path: Path, page_number: int, output: Path) -> None:
    """Render one PDF page in a child process to isolate pypdfium2 native code."""
    worker = _worker_path()
    timeout = max(30, int(os.environ.get("THAMUDIC_PDF_RENDER_TIMEOUT", "120")))
    command = [sys.executable, str(worker), "--render-pdf", str(path), "--page", str(page_number), "--output", str(output)]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, env=_worker_env(), check=False)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"PDF rendering timed out after {timeout}s; the GUI was protected and remains running.") from exc
    except OSError as exc:
        raise RuntimeError(f"Could not start isolated PDF renderer: {exc}") from exc
    if completed.returncode != 0 or not output.exists():
        detail = (completed.stderr or completed.stdout or "unknown PDF rendering error").strip()
        raise RuntimeError(f"PDF page rendering failed safely: {detail[-1000:]}")


def extract_media_text(path: str | Path, ocr_languages: list[str] | None = None) -> tuple[str, dict[str, Any]]:
    p = Path(path).expanduser()
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(p)
    suffix = p.suffix.casefold()
    if suffix == ".pdf":
        text, meta = _pdf_text(p)
        if text:
            return text, {"source_file": str(p), "media_type": "pdf", **meta}
        try:
            from pypdf import PdfReader  # type: ignore
            page_count = len(PdfReader(str(p)).pages)
        except Exception as exc:
            raise RuntimeError(f"Scanned PDF page count could not be read: {exc}") from exc
        all_text: list[str] = []
        confidences: list[float] = []
        errors: list[str] = []
        with tempfile.TemporaryDirectory(prefix="thamudic-pdf-") as tmp:
            for index in range(page_count):
                image_path = Path(tmp) / f"page-{index + 1}.png"
                try:
                    _render_pdf_page(p, index, image_path)
                    page_text, ocr_meta = _easyocr_image(image_path, ocr_languages)
                    if page_text:
                        all_text.append(page_text)
                    if "ocr_confidence" in ocr_meta:
                        confidences.append(ocr_meta["ocr_confidence"])
                except Exception as exc:
                    errors.append(f"page {index + 1}: {type(exc).__name__}: {exc}")
        meta = {
            "source_file": str(p), "media_type": "pdf", "provider": "pypdfium2+easyocr-subprocess",
            "pages": page_count, "ocr_confidence": round(sum(confidences) / len(confidences), 6) if confidences else 0.0,
            "scanned_pages": True, "ocr_errors": errors, "ocr_available": not errors or bool(all_text),
        }
        return "\n".join(all_text).strip(), meta
    if suffix in _IMAGE_SUFFIXES:
        try:
            text, meta = _easyocr_image(p, ocr_languages)
            return text, {"source_file": str(p), "media_type": "image", "ocr_available": True, **meta}
        except Exception as exc:
            return "", {"source_file": str(p), "media_type": "image", "provider": "easyocr-subprocess", "ocr_available": False, "ocr_error": f"{type(exc).__name__}: {exc}"}
    if suffix in _TEXT_SUFFIXES:
        try:
            return p.read_text(encoding="utf-8"), {"source_file": str(p), "media_type": "text", "provider": "utf8"}
        except UnicodeDecodeError as exc:
            raise RuntimeError(f"Text file is not valid UTF-8: {p.name}") from exc
    raise ValueError(f"unsupported media type: {suffix}")


def scan_translate_media(path: str | Path, script: str = "Dadanitic", target_language: str = "en", ocr_languages: list[str] | None = None) -> dict[str, Any]:
    text, media = extract_media_text(path, ocr_languages)
    scan = scan_source_language(text, language="ancient-north-arabian" if script else None)
    result = translate(text, script=script, target_language=target_language) if text else {"translation": None, "translation_status": "not_available", "confidence": 0.0, "transliteration": ""}
    result["media"] = media
    result["scan"] = scan
    result["translation_ready"] = bool(result.get("translation"))
    result["workflow"] = "import -> extract/OCR -> scan -> transliterate -> corpus translation"
    return result
