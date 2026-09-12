# Thamudic Cross-Language Epigraphy Platform

A research-oriented integration layer for the Thamudic/Ancient North Arabian scanner and related ancient-language research workflows. It combines a provenance-aware SQLite/PostgreSQL-compatible data model, FastAPI API, TypeScript/React UI, PDF research exchange, KPI dashboards, Unicode-aware language registries, web asset/source auditing, CSV import/export, and cross-language clients.

## Important source/licensing boundary

The three supplied public deployments are treated as **references and integration targets**, not as permission to republish third-party proprietary bundles. The automated auditor records public HTML/script/style/asset URLs and metadata when a deployment is reachable; it does not silently copy minified third-party bundles into this repository. Implementations in this directory are clean-room equivalents of observed functionality and use only repository-owned code plus permissively licensed/open-source patterns.

The supplied deployments could not be fetched from this execution environment, so `tools/site_audit.py` is included for a reproducible local crawl of:

- `https://thamudicscan-s3wz30.public.builtwithrocket.new/`
- `https://chimera-ii-os-730893.onhercules.app/`
- `https://thamudic-scanner.softr.app/`

## Architecture

```text
Historical PDFs / inscriptions / source records
                    |
             bounded PDF import
                    |
 Unicode/script/language registry + provenance
                    |
   +----------------+----------------+
   |                                 |
FastAPI research API            KPI service
   |                                 |
   +---------------+-----------------+
                   |
          React/TypeScript UI
                   |
 Python | C++ | C# | Java | Go | Rust | JS/TS
                   |
        research PDF export
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
- Parameterized query library and provenance-preserving translation records.
- Softr CSV import/export with stable `Record ID` support.
- Public-site asset audit with robots-aware, bounded crawling.
- Script/style/link manifest with SHA-256 hashes and content types.
- React/TypeScript research UI with upload, search, readings, annotations and export.
- FastAPI JSON API.
- Cross-language REST clients in C++, C#, Java, Go and Rust, with JavaScript/TypeScript integration planned by the common API contract.
- Windows CMD, PowerShell, Bash and Docker deployment/build automation.
- Code/source citations collected in `docs/SOURCES.md`.

## PDF workflows

`POST /api/pdf/import` accepts a bounded PDF and returns page-level extracted text, SHA-256 identity, page count and warnings. OCR is not implicitly mixed into source extraction; OCR/model output can be attached as a separate provenance layer.

`POST /api/pdf/export` produces a research PDF from explicitly supplied object/reading/translation data and records a manifest containing source identity, rights, provenance, citations and PDF hash.

## KPI API

- `GET /api/kpis/summary`
- `GET /api/kpis/languages`

The KPI service is designed as the common contract for application dashboards and language clients; the UI must consume API values rather than hardcoded totals.

## Standards and citations

- Unicode CLDR: https://cldr.unicode.org/
- Unicode BCP 47 extensions: https://cldr.unicode.org/index/bcp47-extension
- Unicode Transliteration Guidelines: https://cldr.unicode.org/index/cldr-spec/transliteration-guidelines
- RFC 6497 transformed-content extension: https://www.rfc-editor.org/rfc/rfc6497
- pypdf: https://github.com/py-pdf/pypdf
- ReportLab: https://www.reportlab.com/
- JSON Schema: https://json-schema.org/

See `docs/SOURCES.md`, `docs/PDF_CITATIONS.md`, `docs/KPI_CITATIONS.md`, and the design/implementation-plan documents under `docs/superpowers/` for detailed source attribution.

## Quick start

### Python API

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .\\.venv\\Scripts\\Activate.ps1
pip install -r server/requirements.txt
uvicorn server.app:app --reload --port 8010
```

### Web UI

```bash
cd web
npm install
npm run dev
```

### Database

```bash
sqlite3 data/thamudic_platform.sqlite < database/schema.sql
sqlite3 data/thamudic_platform.sqlite < database/views.sql
sqlite3 data/thamudic_platform.sqlite < database/migrations/002_pdf_translation_kpi.sql
```

### Audit the supplied sites

```bash
python tools/site_audit.py --out data/site-audit.json
```

Only crawl sites you are authorized to inspect and respect their terms, robots directives, rate limits and copyright/license terms.
