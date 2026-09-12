# Intelligent OCR citations and implementation notes

## Why a pluggable scanner

Historical and non-Latin documents require layout analysis, reading-order handling and recognition models that general-purpose OCR does not necessarily provide. Kraken explicitly targets historical/non-Latin material and supports RTL, BiDi and top-to-bottom writing. Cuneiform research systems demonstrate sign detection, line segmentation and transliteration-aligned evaluation. Tesseract remains useful as a general OCR adapter when appropriate traineddata is installed.

## Implementation

`server/ocr_scanner.py` implements:

- bounded image input;
- SHA-256 source identity;
- image quality metrics (dimensions, mean intensity, contrast and blur proxy);
- pluggable Kraken/Tesseract adapters;
- confidence-based selection in `auto` mode;
- Unicode NFC normalization;
- script-candidate scoring;
- OCR bounding boxes where the engine supplies them;
- warnings for low-quality or unavailable-engine conditions;
- explicit recognition-only provenance.

## Dependency references

As of September 2026, PyPI lists Kraken 7.1.1 as the current release, supporting Python 3.10–3.13 under Apache-2.0. The automation therefore uses the bounded requirement `kraken>=7,<8` when `INSTALL_KRAKEN=1`. Pytesseract 0.3.13 is the Python adapter; the native Tesseract executable and traineddata remain operating-system dependencies. See the project pages below for authoritative installation/licensing information.

- https://pypi.org/project/kraken/
- https://github.com/mittagessen/kraken
- https://pypi.org/project/pytesseract/
- https://github.com/tesseract-ocr/tesseract
- https://github.com/CompVis/cuneiform-sign-detection-code
- https://github.com/ElectronicBabylonianLiterature/cuneiform-ocr
- https://github.com/DigitalPasts/CuReD
- https://www.unicode.org/standard/supported.html
- https://www.unicode.org/versions/Unicode18.0.0/

## Scholarly safety boundary

OCR output is a recognition hypothesis. It is not automatically a transliteration, reconstruction, translation or historical claim. Alternative readings, confidence, model identity and source hashes must remain attached to downstream records.
