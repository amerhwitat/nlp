#!/usr/bin/env python3
"""Thamudic / Ancient North Arabian inscription scanner.

Research-oriented baseline scanner. It deliberately does NOT use Tesseract or
camel_tools. It performs image normalization, foreground segmentation,
connected-component extraction, Unicode-aware metadata, and exports JSON/CSV.
Recognition is intentionally separated from segmentation so trained models can
be added later without changing the ingestion pipeline.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageFilter, ImageOps

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise SystemExit("numpy is required: pip install numpy pillow") from exc

# Unicode Old North Arabian block: U+10A80..U+10A9F.
# Transliteration follows the Unicode character names/list where available.
ONA = {
    0x10A80: "h", 0x10A81: "l", 0x10A82: "ḥ", 0x10A83: "m",
    0x10A84: "q", 0x10A85: "w", 0x10A86: "s2", 0x10A87: "r",
    0x10A88: "b", 0x10A89: "t", 0x10A8A: "s1", 0x10A8B: "k",
    0x10A8C: "n", 0x10A8D: "ḫ", 0x10A8E: "ṣ", 0x10A8F: "s3",
    0x10A90: "f", 0x10A91: "ʾ", 0x10A92: "ʿ", 0x10A93: "ḍ",
    0x10A94: "g", 0x10A95: "d", 0x10A96: "ġ", 0x10A97: "ṭ",
    0x10A98: "z", 0x10A99: "ḏ", 0x10A9A: "y", 0x10A9B: "ṯ",
    0x10A9C: "ẓ", 0x10A9D: "NUMBER_ONE", 0x10A9E: "NUMBER_TEN",
    0x10A9F: "NUMBER_TWENTY",
}

@dataclass
class GlyphBox:
    index: int
    x: int
    y: int
    width: int
    height: int
    area: int
    aspect_ratio: float
    confidence: float
    unicode_candidate: str = ""
    transliteration_candidate: str = ""


def normalize(image: Image.Image, scale: int = 2) -> Image.Image:
    """Convert an inscription photograph into a high-contrast grayscale image."""
    image = ImageOps.exif_transpose(image).convert("L")
    image = ImageOps.autocontrast(image)
    image = image.filter(ImageFilter.MedianFilter(size=3))
    if scale > 1:
        image = image.resize((image.width * scale, image.height * scale), Image.Resampling.LANCZOS)
    return image


def foreground_mask(image: Image.Image, threshold: int = 150) -> np.ndarray:
    """Return a boolean foreground mask using a deterministic threshold."""
    arr = np.asarray(image, dtype=np.uint8)
    # Invert so carved/dark strokes become foreground.
    return arr < threshold


def connected_components(mask: np.ndarray, min_area: int = 25) -> List[Tuple[int, int, int, int, int]]:
    """Extract 8-connected components without OpenCV/Tesseract dependencies."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components = []
    neighbors = ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1))
    for y0 in range(h):
        for x0 in range(w):
            if not mask[y0, x0] or seen[y0, x0]:
                continue
            stack = [(y0, x0)]
            seen[y0, x0] = True
            minx = maxx = x0
            miny = maxy = y0
            area = 0
            while stack:
                y, x = stack.pop()
                area += 1
                minx, maxx = min(minx, x), max(maxx, x)
                miny, maxy = min(miny, y), max(maxy, y)
                for dy, dx in neighbors:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if area >= min_area:
                components.append((minx, miny, maxx + 1, maxy + 1, area))
    return sorted(components, key=lambda b: (b[1], b[0]))


def scan_image(path: Path, threshold: int = 150, min_area: int = 25, scale: int = 2) -> dict:
    image = normalize(Image.open(path), scale=scale)
    mask = foreground_mask(image, threshold)
    boxes = connected_components(mask, min_area)
    glyphs = []
    for i, (x1, y1, x2, y2, area) in enumerate(boxes):
        width, height = x2 - x1, y2 - y1
        # A segmentation confidence, not a recognition probability.
        fill = area / max(width * height, 1)
        confidence = min(1.0, max(0.0, 0.5 * fill + 0.5 * min(width, height) / max(width, height)))
        glyphs.append(GlyphBox(i, x1, y1, width, height, area, round(width / max(height, 1), 4), round(confidence, 4)))
    return {
        "schema": "thamudic-scanner/v1",
        "source_image": str(path),
        "script_family": "Ancient North Arabian / Old North Arabian",
        "unicode_range": "U+10A80-U+10A9F",
        "segmentation": {"threshold": threshold, "min_area": min_area, "scale": scale},
        "recognition_status": "segmentation_only",
        "glyphs": [asdict(g) for g in glyphs],
        "notes": [
            "Thamudic is a scholarly umbrella covering multiple Ancient North Arabian varieties.",
            "A bounding box is not a script identification or translation.",
            "Human/scholarly verification is required before publication of a reading.",
        ],
    }


def export_result(result: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".json":
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    if output.suffix.lower() == ".csv":
        fields = list(result["glyphs"][0].keys()) if result["glyphs"] else ["index"]
        with output.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(result["glyphs"])
        return
    raise ValueError("Output must end in .json or .csv")


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan Ancient North Arabian/Thamudic inscription images")
    parser.add_argument("image", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("thamudic_scan.json"))
    parser.add_argument("--threshold", type=int, default=150)
    parser.add_argument("--min-area", type=int, default=25)
    parser.add_argument("--scale", type=int, default=2)
    args = parser.parse_args()
    result = scan_image(args.image, args.threshold, args.min_area, args.scale)
    export_result(result, args.output)
    print(f"Detected {len(result['glyphs'])} candidate glyph components -> {args.output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
