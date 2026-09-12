# Sources and code citations

## Supplied deployments

1. `https://thamudicscan-s3wz30.public.builtwithrocket.new/` — supplied public Thamudic Scanner deployment.
2. `https://chimera-ii-os-730893.onhercules.app/` — supplied public Chimera II OS deployment.
3. `https://thamudic-scanner.softr.app/` — supplied Softr Thamudic Scanner deployment.

These URLs are integration references. This execution environment returned cache misses for all three, so no claim is made that their private source code or database internals were recovered. Run `tools/site_audit.py` from an authorized network to produce an asset manifest.

## Open-source research and OCR references

- [READ](https://github.com/readsoftware/read) — Research Environment for Ancient Documents; linked images, parallel transcriptions, translations, glossaries and paleographic charts.
- [Cuneiform Detector](https://github.com/marie-saccucci/cuneiform-detector) — open-source inscription-image detection architecture.
- [Textorcist](https://github.com/bhagesh-h/textorcist) — React/TypeScript/Vite OCR UI patterns and export workflows.
- [OpenEtruscan](https://www.openetruscan.com/docs) — computational epigraphy platform and scholarly provenance concepts.
- [Kraken](https://github.com/mittagessen/kraken) — historical/non-Latin OCR/ATR engine with trainable layout, reading-order and recognition models and support for RTL/BiDi/top-to-bottom material.
- [Tesseract](https://github.com/tesseract-ocr/tesseract) — open-source OCR engine with LSTM recognition in the 4.x/5.x generation; used here as an optional adapter, not as the ancient-language source of truth.
- [Cuneiform sign detection code](https://github.com/CompVis/cuneiform-sign-detection-code) — weakly supervised cuneiform sign detection, line segmentation, evaluation and visualization research implementation.
- [Electronic Babylonian Literature cuneiform OCR](https://github.com/ElectronicBabylonianLiterature/cuneiform-ocr) — cuneiform detection/classification/OCR project family and data pipeline references.
- [CuReD](https://github.com/DigitalPasts/CuReD) — OCR for cuneiform printed transliterations using Kraken and specialist models.

The platform implements clean-room adapters and contracts. Third-party source code and model weights are not copied into the repository unless their license and redistribution requirements explicitly permit it.

## Unicode, language identification and transliteration

- [Unicode Supported Scripts](https://www.unicode.org/standard/supported.html) — current encoded-script inventory, including Old North Arabian, Old South Arabian, Cuneiform, Egyptian Hieroglyphs, Linear B, Nabataean and newer Unicode 18.0 additions.
- [Unicode 18.0](https://www.unicode.org/versions/Unicode18.0.0/) — current draft/implementation data used to keep script routing forward-compatible; formal release timing is authoritative on Unicode's site.
- [Unicode CLDR Project](https://cldr.unicode.org/) — locale and language infrastructure used for internationalized software.
- [Unicode BCP 47 Extensions](https://cldr.unicode.org/index/bcp47-extension) — machine-readable BCP 47 extension data.
- [Unicode Transliteration Guidelines](https://cldr.unicode.org/index/cldr-spec/transliteration-guidelines) — distinguishes transliteration from translation and provides transliterator design guidance.
- [RFC 6497](https://www.rfc-editor.org/rfc/rfc6497) — BCP 47 extension for transformed content, including transliteration, transcription and translation.

## PDF implementation references

- [pypdf](https://github.com/py-pdf/pypdf) — Python PDF parsing and text extraction dependency.
- [ReportLab](https://www.reportlab.com/) — PDF generation dependency used for research reports.
- [JSON Schema](https://json-schema.org/) — machine-readable contract validation for PDF/KPI manifests.

## Public Thamudic image references

Wikimedia Commons contains Thamudic inscription imagery under explicit Creative Commons licenses. Dataset ingestion must retain the original license and attribution requirements.

## Existing repository sources

- `python/thamudic/old_north_arabian.py` — canonical Unicode registry/transliteration.
- `ThamudicScan/server/` — existing FastAPI scanner backend.
- `ThamudicScan/web_ui/` — existing React scanner UI.
- `SOFTR_DATABASE_MIGRATION.md` — repository's previously documented Softr migration model.
- `THAMUDIC_SCANNER_RESEARCH.md` — project research notes.
- `ThamudicEpiPlatform/database/migrations/002_pdf_translation_kpi.sql` — PDF/translation/KPI persistence model.
- `ThamudicEpiPlatform/server/pdf_import.py` — bounded PDF extraction and page provenance.
- `ThamudicEpiPlatform/server/pdf_export.py` — provenance-aware research PDF generation.
- `ThamudicEpiPlatform/server/kpi.py` — common KPI aggregation service.
- `ThamudicEpiPlatform/server/ocr_scanner.py` — intelligent OCR routing, quality metrics, engine adapters and provenance.
- `ThamudicEpiPlatform/*/ocr_scan.*` — language-specific OCR API clients.

## Citation policy

Every imported external source should retain URL, retrieval timestamp, license/rights statement, content hash and provenance. Do not redistribute a third-party minified JavaScript bundle merely because it is publicly reachable; reproduce behavior with clean-room code or import it only when its license explicitly permits redistribution.
