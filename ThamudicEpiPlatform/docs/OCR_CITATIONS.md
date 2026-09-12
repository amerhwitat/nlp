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

## Research references

- https://github.com/mittagessen/kraken
- https://github.com/tesseract-ocr/tesseract
- https://github.com/CompVis/cuneiform-sign-detection-code
- https://github.com/ElectronicBabylonianLiterature/cuneiform-ocr
- https://github.com/DigitalPasts/CuReD
- https://www.unicode.org/standard/supported.html
- https://www.unicode.org/versions/Unicode18.0.0/

## Scholarly safety boundary

OCR output is a recognition hypothesis. It is not automatically a transliteration, reconstruction, translation or historical claim. Alternative readings, confidence, model identity and source hashes must remain attached to downstream records.
