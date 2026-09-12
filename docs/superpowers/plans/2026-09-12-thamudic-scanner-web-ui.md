# Thamudic Scanner Web UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production-structured React + Vite and FastAPI web interface for Thamudic/Ancient North Arabian scanning, with upload, live progress, persisted sessions, Unicode-aware results, and CSV/JSON export.

**Architecture:** `ThamudicScan/web_ui` owns the browser application; `ThamudicScan/server` owns HTTP, validation, session persistence, progress streaming, and export. A narrow scanner adapter calls the existing `python/thamudic` API so the canonical Old North Arabian registry and transliteration mapping remain single-sourced.

**Tech Stack:** Python 3, FastAPI, Pydantic, SQLite, pytest, React, Vite, JavaScript/JSX, browser Fetch/EventSource APIs.

**Spec:** `docs/superpowers/specs/2026-09-12-thamudic-scanner-web-ui-design.md`

## Global Constraints

- Preserve the existing `python/thamudic` Unicode and transliteration implementation rather than duplicating its registry.
- Preserve UTF-8 and direct Old North Arabian U+10A80..U+10A9F rendering.
- Keep Arabic and Hebrew output RTL-aware.
- Do not introduce Tesseract or camel_tools as hidden scanner dependencies.
- Bind local development services to localhost by default.
- Uploaded files are untrusted; validate size/type and never execute their contents.
- Use bounded concurrency and ordered per-session progress events.
- Export must support CSV and JSON.
- Update README/documentation and cross-platform startup/install scripts.

---

### Task 1: Establish the backend test contract

**Files:**
- Create: `ThamudicScan/server/tests/test_api.py`
- Create: `ThamudicScan/server/tests/test_scanner_adapter.py`
- Create: `ThamudicScan/server/tests/test_export.py`
- Create: `ThamudicScan/server/tests/conftest.py`

**Interfaces:**
- Consumes the public behavior defined by the design spec.
- Produces executable pytest contracts for `/health`, `/validate`, `/scan`, export, and scanner adaptation.

- [ ] **Step 1: Write failing tests for Unicode validation and transliteration**

```python
def test_validate_returns_old_north_arabian_codepoints(client):
    response = client.post("/validate", json={"text": "𐪀𐪁𐪂"})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 3
    assert body["codepoints"] == [0x10A80, 0x10A81, 0x10A82]
```

- [ ] **Step 2: Run the new validation test and verify the failure is caused by missing server code**

Run: `pytest ThamudicScan/server/tests/test_api.py::test_validate_returns_old_north_arabian_codepoints -v`
Expected: FAIL because the FastAPI application does not yet expose `/validate`.

- [ ] **Step 3: Write failing scanner adapter and export tests**

```python
def test_scanner_adapter_uses_existing_thamudic_api():
    result = scan_text("𐪀𐪁", ["h"])
    assert result["transliteration"] == "hl"


def test_export_csv_has_stable_columns():
    csv_text = export_results_csv([{
        "id": "1", "source": "test", "text": "𐪀",
        "transliteration": "h", "confidence": 0.9
    }])
    assert "id,source,text,transliteration,confidence" in csv_text
```

- [ ] **Step 4: Run those tests and verify they fail for the intended missing interfaces**

Run: `pytest ThamudicScan/server/tests/test_scanner_adapter.py ThamudicScan/server/tests/test_export.py -v`
Expected: FAIL with missing-module/function errors.

- [ ] **Step 5: Commit the red tests**

```bash
git add ThamudicScan/server/tests
git commit -m "test: define Thamudic scanner web contracts"
```

---

### Task 2: Implement the scanner adapter and result models

**Files:**
- Create: `ThamudicScan/server/scanner_adapter.py`
- Create: `ThamudicScan/server/models.py`
- Modify: `ThamudicScan/server/tests/test_scanner_adapter.py`

**Interfaces:**
- Produces `scan_text(text: str, keywords: list[str] | None) -> dict`.
- Produces `validate_text(text: str) -> dict`.
- Produces Pydantic models for scan requests, scan results, sessions, and progress events.

- [ ] **Step 1: Implement the minimum adapter required by the failing tests**

```python
from python.thamudic import extract, transliterate


def scan_text(text: str, keywords: list[str] | None = None) -> dict:
    extracted = extract(text)
    transliterated = transliterate(extracted)
    return {
        "text": extracted,
        "transliteration": transliterated,
        "keywords": keywords or [],
        "confidence": 1.0 if extracted else 0.0,
    }
```

- [ ] **Step 2: Add Unicode validation without duplicating the registry**

```python
from python.thamudic import BY_CHARACTER, is_old_north_arabian


def validate_text(text: str) -> dict:
    chars = [ch for ch in text if is_old_north_arabian(ch)]
    return {
        "count": len(chars),
        "characters": chars,
        "codepoints": [ord(ch) for ch in chars],
        "known": [BY_CHARACTER[ch] for ch in chars],
    }
```

- [ ] **Step 3: Run adapter tests**

Run: `pytest ThamudicScan/server/tests/test_scanner_adapter.py -v`
Expected: PASS.

- [ ] **Step 4: Commit the adapter and models**

```bash
git add ThamudicScan/server/scanner_adapter.py ThamudicScan/server/models.py ThamudicScan/server/tests/test_scanner_adapter.py
git commit -m "feat: add Thamudic scanner adapter"
```

---

### Task 3: Add persistence and export services

**Files:**
- Create: `ThamudicScan/server/db.py`
- Create: `ThamudicScan/server/exporter.py`
- Modify: `ThamudicScan/server/tests/test_export.py`
- Create: `ThamudicScan/server/tests/test_db.py`

**Interfaces:**
- `create_session() -> str`
- `save_result(session_id: str, result: dict) -> str`
- `get_session(session_id: str) -> dict`
- `list_results(session_id: str) -> list[dict]`
- `export_results_csv(results: list[dict]) -> str`
- `export_results_json(results: list[dict]) -> str`

- [ ] **Step 1: Write failing persistence tests**

```python
def test_session_round_trip(database):
    session_id = database.create_session()
    database.save_result(session_id, {"source": "x", "text": "𐪀"})
    session = database.get_session(session_id)
    assert session["id"] == session_id
    assert database.list_results(session_id)[0]["text"] == "𐪀"
```

- [ ] **Step 2: Verify the persistence tests fail**

Run: `pytest ThamudicScan/server/tests/test_db.py -v`
Expected: FAIL because the database service does not yet exist.

- [ ] **Step 3: Implement SQLite schema and CRUD**

Create `sessions` and `results` tables with indexes on `session_id` and timestamps. Store Unicode as UTF-8 text and numeric confidence as a real value.

- [ ] **Step 4: Implement CSV and JSON exporters**

Use Python's `csv` and `json` modules, preserve Unicode with UTF-8 output, and use a stable column order.

- [ ] **Step 5: Run persistence/export tests**

Run: `pytest ThamudicScan/server/tests/test_db.py ThamudicScan/server/tests/test_export.py -v`
Expected: PASS.

- [ ] **Step 6: Commit persistence/export services**

```bash
git add ThamudicScan/server/db.py ThamudicScan/server/exporter.py ThamudicScan/server/tests/test_db.py ThamudicScan/server/tests/test_export.py
git commit -m "feat: persist scanner sessions and export results"
```

---

### Task 4: Build FastAPI endpoints and ordered progress streaming

**Files:**
- Create: `ThamudicScan/server/main.py`
- Create: `ThamudicScan/server/progress.py`
- Create: `ThamudicScan/server/requirements.txt`
- Modify: `ThamudicScan/server/tests/test_api.py`

**Interfaces:**
- `GET /health`
- `POST /validate`
- `POST /scan`
- `POST /scan_file`
- `GET /sessions/{session_id}`
- `GET /sessions/{session_id}/events`
- `GET /export/{session_id}?format=csv|json`

- [ ] **Step 1: Add failing endpoint tests for health, scan, session, and export**

```python
def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_scan_creates_session(client):
    response = client.post("/scan", json={"text": "𐪀𐪁", "keywords": []})
    assert response.status_code == 200
    assert response.json()["session_id"]
    assert response.json()["results"][0]["transliteration"] == "hl"
```

- [ ] **Step 2: Run API tests and verify failure**

Run: `pytest ThamudicScan/server/tests/test_api.py -v`
Expected: FAIL because the FastAPI application is not yet implemented.

- [ ] **Step 3: Implement FastAPI application and CORS configuration**

Default allowed origin: `http://localhost:5173`. Read additional origins from `THAMUDIC_CORS_ORIGINS` without falling back to wildcard CORS.

- [ ] **Step 4: Implement `/scan` and `/validate` using the adapter and persistence services**

The `/scan` response must include `session_id`, `results`, and a summary containing processed count and match count.

- [ ] **Step 5: Implement bounded scan execution and progress events**

Use an `asyncio.Queue` per active session and a bounded worker count from `THAMUDIC_MAX_WORKERS`, defaulting to a small fixed value. Assign monotonically increasing event sequence numbers before enqueueing events. The SSE endpoint emits events in sequence order and terminates when the session reaches a terminal state.

- [ ] **Step 6: Implement upload validation and `/scan_file`**

Allow configurable safe document/image extensions, enforce `THAMUDIC_MAX_UPLOAD_BYTES`, write uploads to a managed temporary directory, and never expose the local path. Delegate extraction to existing repository functionality where supported.

- [ ] **Step 7: Implement session retrieval and CSV/JSON export endpoints**

- [ ] **Step 8: Run the complete backend suite**

Run: `pytest ThamudicScan/server/tests -v`
Expected: PASS.

- [ ] **Step 9: Commit the FastAPI service**

```bash
git add ThamudicScan/server
git commit -m "feat: add FastAPI Thamudic scanner service"
```

---

### Task 5: Build the React + Vite application shell

**Files:**
- Create: `ThamudicScan/web_ui/package.json`
- Create: `ThamudicScan/web_ui/vite.config.js`
- Create: `ThamudicScan/web_ui/index.html`
- Create: `ThamudicScan/web_ui/src/main.jsx`
- Create: `ThamudicScan/web_ui/src/App.jsx`
- Create: `ThamudicScan/web_ui/src/styles.css`

**Interfaces:**
- Browser client calls the FastAPI endpoints under configurable `VITE_API_BASE_URL`, defaulting to `http://127.0.0.1:8000`.
- App state contains scan request, upload state, progress, statistics, results, and active session id.

- [ ] **Step 1: Create package manifest and failing build expectation**

```json
{
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "test": "vitest run"
  }
}
```

- [ ] **Step 2: Run the build before components exist**

Run: `cd ThamudicScan/web_ui && npm install && npm run build`
Expected: FAIL until the application entry point and dependencies are present.

- [ ] **Step 3: Implement the application shell**

Add File, Tools, and Help navigation; a keyword input; scan controls; file picker/drop zone; progress area; statistics area; results area; and export controls.

- [ ] **Step 4: Implement responsive styling**

Use a compact desktop-first layout that remains usable on tablet/mobile widths. Do not hardcode text direction globally; set `dir="rtl"` on Arabic/Hebrew/script-specific fields as appropriate.

- [ ] **Step 5: Run the Vite production build**

Run: `npm run build`
Expected: PASS.

- [ ] **Step 6: Commit the UI shell**

```bash
git add ThamudicScan/web_ui
 git commit -m "feat: add React Vite Thamudic scanner UI"
```

---

### Task 6: Add live scanning, uploads, results, statistics, and exports

**Files:**
- Create: `ThamudicScan/web_ui/src/components/FileDropzone.jsx`
- Create: `ThamudicScan/web_ui/src/components/ProgressPanel.jsx`
- Create: `ThamudicScan/web_ui/src/components/StatsPanel.jsx`
- Create: `ThamudicScan/web_ui/src/components/ResultsTable.jsx`
- Create: `ThamudicScan/web_ui/src/api.js`
- Modify: `ThamudicScan/web_ui/src/App.jsx`
- Modify: `ThamudicScan/web_ui/src/styles.css`

**Interfaces:**
- `scanText(payload) -> Promise<ScanResponse>`
- `scanFile(file) -> Promise<ScanResponse>`
- `validateText(text) -> Promise<ValidationResponse>`
- `subscribeProgress(sessionId, onEvent, onError) -> () => void`
- `exportSession(sessionId, format) -> void`

- [ ] **Step 1: Write failing frontend tests for result rendering and RTL**

```jsx
it('renders Old North Arabian text and transliteration', () => {
  render(<ResultsTable results={[{
    text: '𐪀𐪁', transliteration: 'hl', confidence: 0.91,
    source: 'sample.txt'
  }]} />);
  expect(screen.getByText('𐪀𐪁')).toBeInTheDocument();
  expect(screen.getByText('hl')).toBeInTheDocument();
});
```

- [ ] **Step 2: Run the frontend test and verify failure**

Run: `npm test -- --run`
Expected: FAIL because the component does not yet exist.

- [ ] **Step 3: Implement API helpers and SSE subscription**

Use `fetch()` for HTTP requests and `EventSource` for `/events`. Close the event source when a scan reaches a terminal state or the component unmounts.

- [ ] **Step 4: Implement file upload/drop-zone behavior**

Show selected filename and validation errors; submit with `FormData`; preserve progress and results without blocking the browser UI.

- [ ] **Step 5: Implement progress panel and statistics**

Show current status, processed count, matched count, current source, percentage, and event messages. Use the backend sequence field to ignore stale/out-of-order events.

- [ ] **Step 6: Implement results table**

Show source, historical-script text, transliteration, confidence, language/variant, and code-point details. Render confidence as a recognition-confidence value, not a historical certainty score.

- [ ] **Step 7: Implement CSV/JSON export controls**

Use the session export endpoints and browser download handling.

- [ ] **Step 8: Run frontend tests and production build**

Run: `npm test -- --run && npm run build`
Expected: PASS.

- [ ] **Step 9: Commit the integrated UI behavior**

```bash
git add ThamudicScan/web_ui
 git commit -m "feat: add live scan progress upload results and export"
```

---

### Task 7: Add cross-platform automation and developer configuration

**Files:**
- Create: `ThamudicScan/scripts/install.sh`
- Create: `ThamudicScan/scripts/install.ps1`
- Create: `ThamudicScan/scripts/install.bat`
- Create: `ThamudicScan/scripts/run-server.sh`
- Create: `ThamudicScan/scripts/run-server.ps1`
- Create: `ThamudicScan/scripts/run-server.bat`
- Create: `ThamudicScan/scripts/run-web.sh`
- Create: `ThamudicScan/scripts/run-web.ps1`
- Create: `ThamudicScan/scripts/run-web.bat`
- Create: `ThamudicScan/server/.env.example`

**Interfaces:**
- Install scripts create the Python environment, install `requirements.txt`, run `npm install`, and fail clearly if required runtimes are missing.
- Run scripts start the backend and frontend with documented defaults.

- [ ] **Step 1: Add environment template**

```text
THAMUDIC_CORS_ORIGINS=http://localhost:5173
THAMUDIC_MAX_UPLOAD_BYTES=52428800
THAMUDIC_MAX_WORKERS=4
THAMUDIC_DATABASE=./data/thamudicscan.sqlite3
```

- [ ] **Step 2: Implement Unix installer and runners**

Use `python -m venv .venv`, activate it, install requirements, then run `npm install` in `web_ui`.

- [ ] **Step 3: Implement PowerShell installer and runners**

Use the same dependency sequence and quote paths safely for Windows PowerShell.

- [ ] **Step 4: Implement Windows batch wrappers**

Delegate to PowerShell where practical while preserving clear error messages.

- [ ] **Step 5: Smoke-test the scripts**

Run the installer in a clean temporary checkout/environment and verify that both services can be launched.

- [ ] **Step 6: Commit automation**

```bash
git add ThamudicScan/scripts ThamudicScan/server/.env.example
git commit -m "build: add cross-platform ThamudicScan setup scripts"
```

---

### Task 8: Update documentation and source-code citation index

**Files:**
- Modify: `README.md`
- Modify: `ThamudicScan/README.md`
- Create: `ThamudicScan/docs/ARCHITECTURE.md`
- Create: `ThamudicScan/docs/API.md`
- Create: `ThamudicScan/docs/DEVELOPMENT.md`
- Modify: `docs/superpowers/specs/2026-09-12-thamudic-scanner-web-ui-design.md` only if implementation decisions require clarifying notes

**Interfaces:**
- Documentation must match actual file paths, commands, endpoint names, environment variables, and supported exports.

- [ ] **Step 1: Document backend startup**

```bash
cd ThamudicScan
python -m uvicorn server.main:app --host 127.0.0.1 --port 8000
```

- [ ] **Step 2: Document frontend startup**

```bash
cd ThamudicScan/web_ui
npm install
npm run dev
```

- [ ] **Step 3: Document API request/response examples**

Include JSON examples for `/validate` and `/scan`, SSE event structure, and CSV/JSON export behavior.

- [ ] **Step 4: Add source-code citation entries**

Extend the root README table with the new server, web UI, tests, and documentation paths without removing existing citation entries.

- [ ] **Step 5: Run a documentation path check**

Verify every documented source path exists and every command matches the actual project layout.

- [ ] **Step 6: Commit documentation**

```bash
git add README.md ThamudicScan/README.md ThamudicScan/docs
 git commit -m "docs: document ThamudicScan web architecture and API"
```

---

### Task 9: Full verification and integration review

**Files:**
- Test: `ThamudicScan/server/tests/`
- Test: `ThamudicScan/web_ui/`
- Modify: any implementation file only when verification identifies a real defect

**Interfaces:**
- Final repository must pass backend tests, frontend tests/build, Unicode regression checks, and documented smoke tests.

- [ ] **Step 1: Run the Python test suite**

Run: `pytest python/tests ThamudicScan/server/tests -v`
Expected: PASS with existing Python regression tests preserved.

- [ ] **Step 2: Run the frontend test suite and production build**

Run: `cd ThamudicScan/web_ui && npm test -- --run && npm run build`
Expected: PASS.

- [ ] **Step 3: Run a Unicode smoke test**

Submit `𐪀𐪁𐪂` through `/validate` and `/scan`; verify code points and transliteration remain UTF-8 and are not escaped into replacement characters.

- [ ] **Step 4: Run upload/export smoke tests**

Upload a small UTF-8 text fixture, retrieve the session, export CSV and JSON, and parse both outputs successfully.

- [ ] **Step 5: Review security defaults**

Confirm no wildcard CORS default, no secrets, no arbitrary upload execution, no filesystem path disclosure, and bounded worker/upload limits.

- [ ] **Step 6: Review source-code citations and documentation**

Confirm the root README and ThamudicScan README cite all maintained implementation areas.

- [ ] **Step 7: Commit only after all verification is green**

```bash
git status
git log -5 --oneline
```

Expected: all intended changes committed and working tree clean, with verification results recorded in the final development note.
