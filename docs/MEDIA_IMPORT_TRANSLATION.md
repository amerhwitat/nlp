# Media import → scan → transliteration → translation

The Python Thamudic and general NLP scanner workflows now share `python/thamudic/media_pipeline.py`.

## Supported inputs

- PNG/JPEG/WebP/BMP/TIFF images
- Text PDFs with local `pypdf` extraction
- Scanned PDFs through `pypdfium2` page rendering plus EasyOCR
- TXT/MD/CSV

## Processing pipeline

1. Import the media.
2. Extract embedded PDF text or run OCR for image/scanned content.
3. Preserve the extracted text and OCR provider/confidence.
4. Detect the requested script using the Unicode/source-language registry.
5. Transliterate supported Ancient North Arabian characters.
6. Query the evidence-backed translation corpus/provider.
7. Return translation, status, confidence and provenance.
8. Never turn an OCR guess into an authoritative ancient reading.

## Important limitation

Generic OCR engines are not automatically trained for every ancient script. EasyOCR can recover modern Arabic/Latin and other supported model languages, but reliable recognition of photographed/handwritten Thamudic glyphs requires a model trained on annotated Thamudic images. The application therefore exposes OCR provenance and refuses to fabricate a transliteration or translation when the OCR output is unsupported.

## CLI

```bash
python python/NLPScanner_AllInOne.py inscription.pdf --script Dadanitic --target en
python python/NLPScanner_AllInOne.py inscription.jpg --script Dadanitic --target en
```

## GUI

```bash
python python/NLPScanner_AllInOne.py --gui
python -m thamudic.gui
```

In the Thamudic desktop GUI, **Import image / PDF** automatically extracts/OCRs the source and immediately runs the transliteration/translation workflow.
