# Thamudic Cross-Language Epigraphy Platform

A research-oriented integration layer for the Thamudic/Ancient North Arabian scanner and related public web applications. It combines a provenance-aware SQLite/PostgreSQL-compatible data model, FastAPI API, TypeScript/React UI, web asset/source auditor, CSV import/export, and language clients.

## Important source/licensing boundary

The three supplied public deployments are treated as **references and integration targets**, not as permission to republish third-party proprietary bundles. The automated auditor records public HTML/script/style/asset URLs and metadata when a deployment is reachable; it does not silently copy minified third-party bundles into this repository. Implementations in this directory are clean-room equivalents of observed functionality and use only repository-owned code plus permissively licensed/open-source patterns.

The supplied deployments could not be fetched from this execution environment, so `tools/site_audit.py` is included for a reproducible local crawl of:

- `https://thamudicscan-s3wz30.public.builtwithrocket.new/`
- `https://chimera-ii-os-730893.onhercules.app/`
- `https://thamudic-scanner.softr.app/`

## Architecture

```text
Public deployments / local source exports
              |
       site_audit.py
              |
      source manifest + rights
              |
   +----------+-----------+
   |                      |
FastAPI + SQLite      React + TypeScript
   |                      |
   +------ SQL schema ----+
              |
     provenance / readings
              |
 Python | C++ | C# | Java | Go | Rust
```

## Features

- UTF-8 Old North Arabian/Thamudic support (`U+10A80-U+10A9F`).
- Objects, annotations, readings, sources and periods tables.
- SQL views for object summaries, confidence dashboards and provenance.
- Parameterized query library.
- Softr CSV import/export with stable `Record ID` support.
- Public-site asset audit with robots-aware, bounded crawling.
- Script/style/link manifest with SHA-256 hashes and content types.
- React/TypeScript research UI with upload, search, readings, annotations and export.
- FastAPI JSON API.
- Cross-language REST clients in C++, C#, Java, Go and Rust.
- Windows CMD, PowerShell, Bash and Docker deployment automation.
- Code/source citations collected in `docs/SOURCES.md`.

## Open-source inspirations

The architecture was informed by public projects such as READ, an open research environment for ancient documents, and modern React/FastAPI OCR/epigraphy applications. The repository does not copy their source code. See `docs/SOURCES.md`.

## Quick start

### Python API

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
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
```

### Audit the supplied sites

```bash
python tools/site_audit.py --out data/site-audit.json
```

Only crawl sites you are authorized to inspect and respect their terms, robots directives, rate limits and copyright/license terms.
