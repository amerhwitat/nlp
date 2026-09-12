from __future__ import annotations

"""Pluggable intelligent OCR scanner for historical/ancient-script material."""

import hashlib
import io
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import cv2
    import numpy as np
except Exception:
    cv2 = np = None

SCRIPT_RANGES = {
    "Old North Arabian": (0x10A80, 0x10A9F), "Old South Arabian": (0x10A60, 0x10A7F),
    "Nabataean": (0x10880, 0x108AF), "Phoenician": (0x10900, 0x1091F),
    "Imperial Aramaic": (0x10840, 0x1085F), "Cuneiform": (0x12000, 0x123FF),
    "Egyptian Hieroglyphs": (0x13000, 0x1342F), "Coptic": (0x2C80, 0x2C9F),
    "Linear B": (0x10000, 0x1007F), "Greek": (0x0370, 0x03FF),
    "Latin": (0x0041, 0x024F), "Hebrew": (0x0590, 0x05FF), "Arabic": (0x0600, 0x06FF),
}

@dataclass
class OCRBox:
    x: int
    y: int
    width: int
    height: int
    text: str = ""
    confidence: float = 0.0

@dataclass
class OCRResult:
    engine: str
    text: str
    confidence: float
    script_candidates: list[dict[str, Any]]
    boxes: list[OCRBox]
    image_quality: dict[str, float]
    warnings: list[str]
    source_sha256: str
    preprocessing: list[str]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["boxes"] = [asdict(x) for x in self.boxes]
        return data

def _quality(raw: bytes) -> dict[str, float]:
    if Image is None:
        return {"width": 0.0, "height": 0.0, "mean": 0.0, "contrast": 0.0, "blur": 0.0}
    image = Image.open(io.BytesIO(raw)).convert("L")
    if np is None:
        return {"width": float(image.width), "height": float(image.height), "mean": 0.0, "contrast": 0.0, "blur": 0.0}
    arr = np.asarray(image, dtype=np.float32)
    blur = float(cv2.Laplacian(arr, cv2.CV_64F).var()) if cv2 is not None else 0.0
    return {"width": float(image.width), "height": float(image.height), "mean": float(arr.mean()), "contrast": float(arr.std()), "blur": blur}

def _script_scores(text: str) -> list[dict[str, Any]]:
    counts = {name: 0 for name in SCRIPT_RANGES}
    for ch in text:
        cp = ord(ch)
        for name, (lo, hi) in SCRIPT_RANGES.items():
            if lo <= cp <= hi:
                counts[name] += 1
    total = sum(counts.values())
    return [{"script": name, "score": round(count / total, 4)} for name, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True) if count] if total else []

def _run_tesseract(raw: bytes, lang: str | None):
    try:
        import pytesseract
        from pytesseract import Output
        if Image is None:
            raise RuntimeError("Pillow is not installed")
        kwargs = {"config": "--psm 6", "output_type": Output.DICT}
        if lang:
            kwargs["lang"] = lang
        data = pytesseract.image_to_data(Image.open(io.BytesIO(raw)), **kwargs)
        words, boxes, confs = [], [], []
        for i, value in enumerate(data.get("text", [])):
            text = str(value).strip()
            if not text:
                continue
            raw_conf = str(data["conf"][i]).strip()
            conf = float(raw_conf) if raw_conf not in {"", "-1"} else 0.0
            score = max(0.0, min(1.0, conf / 100.0))
            boxes.append(OCRBox(int(data["left"][i]), int(data["top"][i]), int(data["width"][i]), int(data["height"][i]), text, score))
            words.append(text); confs.append(score)
        return " ".join(words), (sum(confs) / len(confs) if confs else 0.0), boxes, "tesseract"
    except Exception as exc:
        return "", 0.0, [], f"tesseract unavailable: {exc}"

def _run_kraken(raw: bytes, model: str | None):
    if not model:
        return "", 0.0, [], "kraken model not configured"
    try:
        from kraken import binarization, pageseg, rpred
        from kraken.lib import models
        if Image is None:
            raise RuntimeError("Pillow is not installed")
        image = Image.open(io.BytesIO(raw)).convert("L")
        bw = binarization.nlbin(image); bounds = pageseg.segment(bw); net = models.load_any(model)
        records = rpred.rpred(net, bw, bounds)
        text = "\n".join(record.prediction for record in records)
        confs = [float(record.confidence) for record in records if hasattr(record, "confidence")]
        return text, (sum(confs) / len(confs) if confs else 0.0), [], "kraken"
    except Exception as exc:
        return "", 0.0, [], f"kraken unavailable: {exc}"

def _normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text).replace("\r\n", "\n").strip()

def scan_image(raw: bytes, *, engine: str = "auto", tesseract_lang: str | None = None, kraken_model: str | None = None, max_bytes: int = 25 * 1024 * 1024) -> OCRResult:
    if not raw:
        raise ValueError("empty image")
    if len(raw) > max_bytes:
        raise ValueError(f"image exceeds {max_bytes} byte limit")
    sha = hashlib.sha256(raw).hexdigest(); quality = _quality(raw); warnings: list[str] = []
    if quality["width"] and (quality["width"] < 300 or quality["height"] < 300):
        warnings.append("low-resolution source; recognition confidence may be reduced")
    if quality["contrast"] and quality["contrast"] < 20:
        warnings.append("low contrast source; adaptive preprocessing is recommended")
    if quality["blur"] and quality["blur"] < 50:
        warnings.append("soft/blurred source; sign boundaries may be unreliable")
    candidates = []
    if engine in {"auto", "kraken"}:
        kt, kc, kb, ke = _run_kraken(raw, kraken_model)
        if kt: candidates.append((kc, kt, kb, "kraken"))
        elif engine == "kraken": warnings.append(ke)
    if engine in {"auto", "tesseract"}:
        tt, tc, tb, te = _run_tesseract(raw, tesseract_lang)
        if tt: candidates.append((tc, tt, tb, "tesseract"))
        elif engine == "tesseract": warnings.append(te)
    if candidates:
        confidence, text, boxes, selected = max(candidates, key=lambda item: item[0])
        text = _normalize_text(text)
        preprocessing = ["UTF-8/NFC output normalization", f"selected {selected} result by confidence"]
    else:
        confidence, text, boxes, selected = 0.0, "", [], "quality-only"
        preprocessing = ["UTF-8/NFC output normalization"]
        warnings.append("no OCR model/engine produced text; returning image-quality and routing metadata")
    return OCRResult(selected, text, round(float(confidence), 4), _script_scores(text), boxes, quality, warnings, sha, preprocessing)
