# Thamudic Scanner Web API

The browser client talks to the FastAPI service in `ThamudicScan/server`.

## Start

```bash
python -m pip install -r ThamudicScan/server/requirements.txt
python ThamudicScan/server/run.py
```

The service binds to `127.0.0.1:8000` by default.

## Endpoints

- `GET /health` — service status and version.
- `POST /validate` — return recognized Old North Arabian characters, code points and canonical metadata.
- `POST /scan` — scan JSON `{text, keywords, source}` and create a persisted session.
- `POST /scan_file` — scan a bounded UTF-8 text upload.
- `GET /sessions/{session_id}` — reopen session state, results and progress history.
- `GET /sessions/{session_id}/events` — ordered Server-Sent Events progress stream.
- `GET /export/{session_id}?format=csv|json` — download session results.

## Unicode

The canonical Old North Arabian range is U+10A80–U+10A9F. The web layer imports the registry and transliteration API from `python/thamudic`; it does not duplicate the mapping table.

## Configuration

- `THAMUDIC_HOST` — default `127.0.0.1`.
- `THAMUDIC_PORT` — default `8000`.
- `THAMUDIC_DB_PATH` — SQLite database path.
- `THAMUDIC_CORS_ORIGINS` — comma-separated allowed browser origins.
- `THAMUDIC_MAX_UPLOAD_BYTES` — default 10 MiB.
- `THAMUDIC_ALLOWED_EXTENSIONS` — comma-separated safe extensions; defaults to UTF-8 text/structured-text formats.

Uploads are treated as untrusted data, never executed, and their local filesystem paths are not returned by the API.

## Frontend

```bash
cd ThamudicScan/web_ui
npm install
npm run dev
```

Set `VITE_API_BASE_URL` if the API is hosted somewhere other than `http://127.0.0.1:8000`.
