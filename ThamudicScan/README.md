# ThamudicScan Web Application

## Browser interface

Public deployment:

https://thamudicscan-s3wz30.public.builtwithrocket.new/

The repository now contains a first-party React + Vite browser interface and FastAPI service under `ThamudicScan/web_ui` and `ThamudicScan/server`.

## Local development

### Backend

```bash
python -m pip install -r ThamudicScan/server/requirements.txt
python ThamudicScan/server/run.py
```

Default API: `http://127.0.0.1:8000`.

### Frontend

```bash
cd ThamudicScan/web_ui
npm install
npm run dev
```

Default UI: `http://127.0.0.1:5173`.

Windows users can use the installers and launchers in `ThamudicScan/scripts/`.

## Features

- Keyword-aware text scanning.
- UTF-8 inscription/document upload with bounded file handling.
- Live Server-Sent Events progress.
- Persisted, reopenable SQLite sessions.
- Unicode validation for U+10A80–U+10A9F.
- Canonical extraction/transliteration through `python/thamudic`.
- Results with source, historical-script text, transliteration, recognition confidence and code points.
- CSV and JSON export.
- Arabic/Hebrew-compatible presentation boundaries and direct Old North Arabian Unicode rendering.

Recognition confidence is a scanner/model measure and must not be interpreted as historical certainty.

## API

See `docs/WEB_API.md` for endpoint contracts and configuration. See `docs/ARCHITECTURE.md` for the service boundary and persistence/progress design.

## Security boundary

Development binds to localhost. CORS is explicitly configured rather than wildcarded. Uploaded filenames are treated as untrusted metadata, uploads are size/type bounded, content is decoded as UTF-8 text, and uploaded files are never executed. Future network-source crawling must remain explicitly bounded and rate-limited.

## Repository relationship

The public deployment is an external deployment associated with the research tooling in this repository. Source code, data, and licensing remain governed by the repository files and licenses.
