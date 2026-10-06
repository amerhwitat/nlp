# NLP / Ancient Script Studio

A professional, end-to-end web application that lets a user **upload or photograph
an inscription image, run OCR, then view its transliteration and translation**.
It reuses the existing [`python/thamudic`](../python/thamudic) modules unchanged
(OCR media pipeline, resilient OCR worker, ancient translation corpus, universal
translation facade, script summaries, translation log, PDF export, and the ancient
alphabet registry).

This directory is fully self-contained: existing repository code is never modified.

> Scope: OCR + language translation only. No cryptocurrency / wallet / key
> functionality is added, modified, or referenced here.

---

## Features

- **Backend (FastAPI)** — health, image/PDF upload → OCR → save result,
  translation (offline corpus + an *optional* remote provider hook enabled only
  when an API key is set), script summary, durable translation history, and export
  as JSON / Markdown / TXT / PDF.
- **Frontend (single-page app)** — dark, professional, responsive UI with
  drag-and-drop, file picker, camera capture, an upload/processing progress view,
  and a results panel showing original characters, Unicode/script metadata,
  transliteration, translation, confidence, provider, provenance, copy buttons,
  text-to-speech playback, and export buttons.
- **Static variant** — the same SPA works with **no backend** using Tesseract.js
  for OCR and bundled JSON lookup tables (generated from the Python registry) for
  transliteration, script detection, and offline corpus translation. A GitHub
  Actions workflow publishes it to `https://amerhwitat.github.io/nlp/`.
- **Ops** — Dockerfile + docker-compose, install/run scripts (sh + bat), and an
  automated test suite for the backend and the JSON export.

---

## Quick start

### Backend + frontend (served together)

```bash
cd nlp-studio
./install.sh          # or: install.bat  (Windows)
./run.sh              # or: run.bat
# open http://localhost:8000
```

The API docs (Swagger UI) are at `http://localhost:8000/docs`.

### Docker

```bash
cd nlp-studio
docker compose up --build
# open http://localhost:8000
```

### Static (no-backend) build

The static app already lives in [`web/`](web/) with the generated data tables
committed under [`web/data/`](web/data/). To regenerate the tables from the
Python sources (keep them in sync):

```bash
python nlp-studio/scripts/export_static_data.py
```

Then open `nlp-studio/web/index.html` directly, or serve it with any static host:

```bash
python3 -m http.server 8080 --directory nlp-studio/web
# open http://localhost:8080
```

The app auto-detects whether a backend is reachable; with none it runs fully
client-side.

---

## Reusing `python/thamudic`

The backend adds the repository `python/` directory (and repo root) to `sys.path`
in [`backend/config.py`](backend/config.py) and imports the existing modules as
`python.thamudic.*` — the same convention used by the repository's own test suite.
Nothing under `python/thamudic` is modified.

| Existing module | How it is used |
|---|---|
| `media_pipeline.py` | `extract_media_text` — primary OCR/text-extraction path (pypdf + EasyOCR-in-subprocess). |
| `resilient_ocr.py` | Provides the isolated, crash-safe native OCR design the service follows. |
| `ancient_translation.py` | `translate` — deterministic OCIANA seed corpus + transliteration. |
| `universal_translation.py` | Provider-oriented facade/`TranslationResult` (kept for compatibility). |
| `script_summary.py` | `build_script_summary` / `export_script_summary` for script metadata + export. |
| `translation_log.py` | Durable, hash-verified translation history + `export_records`. |
| `pdf_export.py` | `report_pdf_bytes` for PDF export. |
| `ancient_alphabet_registry.py` | Language profiles / capabilities surfaced to the UI. |

---

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Service + OCR/provider capability status. |
| `POST` | `/api/upload` | `multipart/form-data` image/PDF → OCR → save result. |
| `POST` | `/api/translate` | JSON `{text, script, source_language, target_language}` → transliteration + translation. |
| `GET` | `/api/languages` | Supported scripts, source languages, targets. |
| `GET` | `/api/script-summary/{language}` | Script metadata (add `?format=json\|md\|txt\|pdf`). |
| `GET` | `/api/history` | Translation history + integrity check. |
| `GET` | `/api/export/history?format=json\|jsonl\|txt\|pdf` | Export the history. |
| `GET` | `/api/export/result/{id}?format=json\|md\|txt\|pdf` | Export a saved result. |

### Upload example

```bash
curl -s -X POST http://localhost:8000/api/upload \
  -F "file=@inscription.png" \
  -F "script=Dadanitic" \
  -F "source_language=ancient-north-arabian" \
  -F "target_language=en"
```

### Translate example (text)

```bash
curl -s -X POST http://localhost:8000/api/translate \
  -H "Content-Type: application/json" \
  -d '{"text":"ytm bn ʿbny w wgm ʿl- ḫll -h","script":"Safaitic","source_language":"ancient-north-arabian","target_language":"en"}'
```

---

## Optional remote translation provider hook

The offline OCIANA seed corpus is always available and is preferred. When the
corpus has no matching reading, an optional remote provider is consulted **only if**
`ANCIENT_TRANSLATION_API_KEY` is set in the environment:

```bash
export ANCIENT_TRANSLATION_API_KEY="sk-..."
export ANCIENT_TRANSLATION_API_URL=""     # any OpenAI-compatible endpoint (optional)
export ANCIENT_TRANSLATION_MODEL=""       # model id (optional)
```

Leave the key blank for a fully offline application. Provider failures never break
a request — they degrade to `translation_status: "not_available"`.

---

## Security & uploads

- Uploads are bounded by **size** (10 MB default, `NLP_STUDIO_MAX_UPLOAD_BYTES`)
  and **type** (PNG/JPG/WEBP/BMP/TIFF/PDF), validated by extension **and** magic
  bytes.
- Uploaded files are stored under a generated UUID name and are **never executed,
  imported, or interpreted as code** — they are only handed to image/PDF libraries
  as data.
- CORS is configured explicitly (see `NLP_STUDIO_CORS_ORIGINS`); credentials are
  not allowed cross-origin.

---

## Tests

```bash
cd /path/to/repo          # repo root, so python/thamudic resolves
python -m pytest nlp-studio/backend/tests -q
```

The suite covers health, upload→OCR→save, translation, history/export, and the
static JSON export script. An OCR smoke test generates a real image, runs it
through the pipeline, and verifies the saved result.

---

## Project layout

```
nlp-studio/
├── backend/            FastAPI service (config, ocr_service, translation_service, storage, main)
│   └── tests/          automated backend + export tests
├── scripts/
│   └── export_static_data.py   exports registry/corpus → web/data/*.json
├── web/                single-page app (served by FastAPI AND the static Pages build)
│   ├── index.html / app.js / styles.css
│   └── data/           generated JSON lookup tables (committed, keep in sync)
├── Dockerfile
├── docker-compose.yml
├── install.sh / install.bat
├── run.sh / run.bat
├── requirements.txt
├── .env.example
└── README.md
```

The GitHub Actions workflow `.github/workflows/nlp-studio-pages.yml` (in the repo
root) regenerates the static tables and publishes `web/` to GitHub Pages.
