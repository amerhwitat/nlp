# Thamudic Cross-Language Epigraphy Platform

A research-oriented integration layer for the Thamudic/Ancient North Arabian scanner and related ancient-language research workflows. It combines a provenance-aware SQLite/PostgreSQL-compatible data model, FastAPI API, TypeScript/React UI, PDF research exchange, KPI dashboards, Unicode-aware language registries, intelligent OCR adapters, web asset/source auditing, CSV import/export, and cross-language clients.

## Important source/licensing boundary

The three supplied public deployments are treated as **references and integration targets**, not as permission to republish third-party proprietary bundles. The automated auditor records public HTML/script/style/asset URLs and metadata when a deployment is reachable; it does not silently copy minified third-party bundles into this repository. Implementations in this directory are clean-room equivalents of observed functionality and use only repository-owned code plus permissively licensed/open-source patterns.

## Architecture

```text
Historical PDFs / inscriptions / source records
                    |
        bounded import + image quality checks
                    |
     intelligent OCR engine/router layer
       |             |             |
     Kraken      Tesseract     quality-only
       |             |             |
       +------ confidence/provenance ------+
                    |
 Unicode/script/language registry + review
                    |
   +----------------+----------------+
   |                                 |
FastAPI research API            KPI service
   |                                 |
   +---------------+-----------------+
                   |
          React/TypeScript UI
                   |
 Python | C++ | C# | Java | Go | Rust | JS | TS
                   |
        transliteration / translation / PDF
```

## Features

- UTF-8 Old North Arabian/Thamudic support (`U+10A80-U+10A9F`).
- Ancient-language registry architecture for Mesopotamia, Egypt, Arabia, Greek and Latin.
- Unicode code-point identity kept separate from UTF-8 interchange encoding.
- Objects, annotations, readings, sources and periods tables.
- Provenance-aware PDF import with bounded page/byte limits and page-level extraction.
- Research PDF export for historical objects, scripts, transliteration, literal/meaning translations, confidence and citations.
- Machine-readable PDF manifest and KPI JSON schemas.
- SQL views and API endpoints for application KPIs.
- KPI dimensions for objects, readings, review, translation confidence, PDF jobs, errors and processing performance.
- **Intelligent OCR scanner** with image SHA-256 provenance, resolution/contrast/blur quality metrics, pluggable Kraken/Tesseract adapters, confidence-based engine selection, Unicode NFC normalization, script candidate scoring and OCR bounding boxes when supplied by the engine.
- OCR recognition is explicitly separate from transliteration and translation; low-confidence/engine failures are surfaced as warnings rather than fabricated text.
- Optional Kraken installation for historical/non-Latin material; optional Tesseract adapter for installed traineddata.
- Parameterized query library and provenance-preserving translation records.
- Softr CSV import/export with stable `Record ID` support.
- Public-site asset audit with robots-aware, bounded crawling.
- React/TypeScript research UI with search, KPI cards, image OCR scanning, readings, annotations and export.
- FastAPI JSON API.
- Cross-language OCR/API clients in Python, C++, C#, Java, Go, Rust, JavaScript and TypeScript.
- Windows CMD, PowerShell, Bash and Docker deployment/build automation with dependency checks.
- Code/source citations collected in `docs/SOURCES.md`.

## OCR workflows

`POST /api/ocr/scan` accepts a bounded image upload and returns a recognition result, confidence, script candidates, bounding boxes, image-quality metrics, warnings, preprocessing steps and SHA-256 provenance.

Supported engine modes:

- `auto`: choose the highest-confidence configured result.
- `kraken`: historical/non-Latin OCR adapter; a compatible model must be configured with `kraken_model`.
- `tesseract`: general OCR adapter using installed Tesseract traineddata.
- `quality-only`: use the scanner as a source-quality/provenance gate without recognition.

The scanner does not assert that an OCR result is a scholarly transliteration or translation. Reviewers must preserve alternatives and uncertainty in the reading layer.

### Dependency automation

```bash
# Linux/macOS
./scripts/check-dependencies.sh

# Windows PowerShell
./scripts/check-dependencies.ps1

# Windows CMD
scripts\\check-dependencies.bat
```

Set `INSTALL_KRAKEN=1` when the optional Kraken package should be installed automatically. Tesseract is an operating-system binary and is therefore detected rather than silently installed by Python package management.

## PDF workflows

`POST /api/pdf/import` accepts a bounded PDF and returns page-level extracted text, SHA-256 identity, page count and warnings. OCR is not implicitly mixed into source extraction; OCR/model output can be attached as a separate provenance layer.

`POST /api/pdf/export` produces a research PDF from explicitly supplied object/reading/translation data and records a manifest containing source identity, rights, provenance, citations and PDF hash.

## KPI API

- `GET /api/kpis/summary`
- `GET /api/kpis/languages`

The KPI service is designed as the common contract for application dashboards and language clients; the UI consumes API values rather than hardcoded totals.

## Standards and citations

- Unicode supported scripts: https://www.unicode.org/standard/supported.html
- Unicode 18.0 implementation/draft data: https://www.unicode.org/versions/Unicode18.0.0/
- Unicode CLDR: https://cldr.unicode.org/
- Unicode BCP 47 extensions: https://cldr.unicode.org/index/bcp47-extension
- Unicode Transliteration Guidelines: https://cldr.unicode.org/index/cldr-spec/transliteration-guidelines
- RFC 6497 transformed-content extension: https://www.rfc-editor.org/rfc/rfc6497
- Kraken OCR: https://github.com/mittagessen/kraken
- Tesseract OCR: https://github.com/tesseract-ocr/tesseract
- Cuneiform sign-detection research implementation: https://github.com/CompVis/cuneiform-sign-detection-code
- Electronic Babylonian Literature cuneiform OCR: https://github.com/ElectronicBabylonianLiterature/cuneiform-ocr
- CuReD: https://github.com/DigitalPasts/CuReD
- pypdf: https://github.com/py-pdf/pypdf
- ReportLab: https://www.reportlab.com/
- JSON Schema: https://json-schema.org/

See `docs/SOURCES.md`, `docs/PDF_CITATIONS.md`, `docs/KPI_CITATIONS.md`, and the design/implementation-plan documents under `docs/superpowers/` for detailed source attribution.

## Quick start

```bash
cd ThamudicEpiPlatform
./scripts/check-dependencies.sh
source .venv/bin/activate
python -m uvicorn server.app:app --reload --port 8010
```

For Windows use `scripts\\check-dependencies.ps1` or `scripts\\check-dependencies.bat` first, then start `server.app` with the `.venv` interpreter.

Only crawl sites you are authorized to inspect and respect their terms, robots directives, rate limits and copyright/license terms.
