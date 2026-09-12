# Thamudic Scanner Web UI Design

**Date:** 2026-09-12
**Status:** Approved design

## Goal

Add a modern browser-based Thamudic/Ancient North Arabian research scanner to the `nlp` repository, using a React + Vite frontend and FastAPI backend while preserving and reusing the repository's existing Unicode, UTF-8, transliteration, and scanner logic.

## Scope

The web application provides:

- Keyword-driven Thamudic/Ancient North Arabian scanning.
- Text input and inscription/document/image upload.
- Live scan progress with current URL/source, processed counts, and status.
- Results table/grid with source, detected text, transliteration, confidence, and metadata.
- Unicode-aware Thamudic rendering and RTL-aware Arabic/Hebrew presentation.
- Validation of supplied text and Unicode code points.
- CSV and JSON result export.
- SQLite-backed scan/session persistence and resumable session state.
- Local development and cross-platform startup/install scripts.
- API and UI tests plus Unicode regression coverage.

## Existing implementation boundary

The canonical Python API already exposes Old North Arabian extraction and transliteration. `python/thamudic/__init__.py` provides `extract()` and `transliterate()`, while `python/thamudic/old_north_arabian.py` defines the U+10A80..U+10A9F registry, UTF-8 bytes, code-point metadata, and transliteration mapping. The new server layer will adapt these APIs instead of duplicating their mapping tables.

The repository also documents a `ThamudicScan/` web-application area. The new implementation will live there so existing research documentation remains grouped with the web product.

## Architecture

```text
Browser
  |
  v
React + Vite (`ThamudicScan/web_ui`)
  |
  | HTTP + Server-Sent Events
  v
FastAPI (`ThamudicScan/server`)
  |
  +--> Scanner adapter --> `python/thamudic`
  |
  +--> File/text extraction boundary
  |
  +--> SQLite session/result store
  |
  +--> CSV/JSON exporter
```

### Frontend

React + Vite is preferred over the legacy Create React App/react-scripts approach. The UI will be componentized around file upload, scan controls, progress, statistics, results, and export actions.

### Backend

FastAPI will expose small, explicit endpoints:

- `GET /health` — service health and version information.
- `POST /scan` — scan submitted text/keywords.
- `POST /scan_file` — scan an uploaded supported file.
- `POST /validate` — validate text and return recognized Old North Arabian code points.
- `GET /sessions/{session_id}` — retrieve persisted session state and results.
- `GET /sessions/{session_id}/events` — stream progress events using SSE.
- `GET /export/{session_id}?format=csv|json` — export results.

The backend will use a scanner adapter so HTTP concerns do not leak into the Python language-processing package.

## Scan result model

Each result should contain a stable identifier and structured fields equivalent to:

```json
{
  "id": "result-id",
  "session_id": "session-id",
  "source": "uploaded-file-or-source",
  "text": "𐪀𐪁𐪂",
  "transliteration": "hlm",
  "confidence": 0.97,
  "language": "Old North Arabian",
  "script_variant": "Dadanitic",
  "codepoints": [68224, 68225, 68226]
}
```

Confidence values are scanner/model outputs, not claims of historical certainty. The UI must label them as recognition confidence.

## File handling

Uploads will be bounded by configurable size limits and stored only in an application-managed temporary/session area. Filenames are treated as untrusted metadata. Supported document/image processing will use the repository's existing extraction/classification capabilities where available; the web layer will not introduce Tesseract or camel_tools as hidden dependencies.

## Unicode and RTL requirements

- Preserve UTF-8 end-to-end.
- Render Old North Arabian characters directly from Unicode U+10A80..U+10A9F.
- Preserve Arabic and Hebrew text direction where present.
- Expose code point and UTF-8 metadata in validation/result details.
- Never replace historical-script characters with ASCII approximations in the primary result field.

## Progress and concurrency

The backend may process independent scan units concurrently, but progress updates must remain ordered per session. The default implementation will use bounded worker concurrency and an event queue so a large scan cannot create an unbounded number of tasks. The frontend will consume SSE events and update progress without blocking the results view.

## Security and operational boundaries

- Bind development services to localhost by default.
- Make allowed CORS origins configurable.
- Validate uploaded file type/size before processing.
- Never execute uploaded content.
- Do not expose filesystem paths in API responses.
- Do not store secrets in source control.
- Keep network crawling, if later connected to external sources, behind explicit bounded/rate-limited interfaces rather than unrestricted server-side fetching.

## Testing

Backend tests will cover:

1. Health endpoint.
2. Unicode validation and U+10A80..U+10A9F handling.
3. Text scanning and transliteration integration.
4. File upload validation.
5. Session persistence/resume.
6. CSV/JSON export.
7. SSE progress event ordering.

Frontend validation will include a production Vite build and component-level tests for upload, progress, result rendering, RTL text, and export controls.

## Documentation

Update:

- Root `README.md` with the new web architecture and source-code citation index.
- `ThamudicScan/README.md` with setup, development, API, export, and deployment notes.
- `ThamudicScan/docs/` with API and architecture documentation.
- Cross-platform install/run scripts and dependency manifests.

## Acceptance criteria

The feature is complete when a fresh checkout can start the FastAPI service and Vite UI using documented commands; a user can submit Thamudic text or an accepted file, observe live progress, inspect Unicode/transliteration/confidence results, persist and reopen a session, and export its results as CSV or JSON; automated tests cover the backend behavior and the frontend production build succeeds.
