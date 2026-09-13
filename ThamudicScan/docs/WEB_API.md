# Thamudic / Ancient Script Scanner Web API

The browser client talks to the FastAPI service in `ThamudicScan/server`.

## Start

```bash
python -m pip install -r ThamudicScan/server/requirements.txt
python ThamudicScan/server/run.py
```

The service binds to `127.0.0.1:8000` by default.

## Endpoints

- `GET /health` — service status and version.
- `GET /alphabet-languages` — all registered ancient/classical language IDs.
- `GET /alphabet-languages/{language}` — script, historical variants, Unicode blocks and translation directions.
- `GET /translation-matrix` — translation capability matrix.
- `POST /scan_language` — source-language Unicode/UTF-8 scan.
- `POST /validate` — recognized Old North Arabian characters, code points and canonical metadata.
- `POST /translate` — audited Ancient North Arabian baseline translation.
- `POST /translate_ancient` — universal provider-facing translation contract.
- `GET /script-summary/{language}` — complete metadata summary for one script/language.
- `GET /script-summary/{language}/export?format=json|md|txt` — metadata export.
- `POST /script-report` — combine exact original text with script metadata, transliteration and translation result.
- `POST /script-report/export` — export the combined report as JSON, Markdown or TXT.
- `GET /voice/capabilities` — available TTS/STT capabilities and voice-control vocabulary.
- `POST /voice/speak` — request a voice-provider playback capability response.
- `POST /scan` — scan JSON `{text, keywords, source}` and create a persisted session.
- `POST /scan_file` — scan a bounded UTF-8 text upload.
- `GET /sessions/{session_id}` — reopen session state, results and progress history.
- `GET /sessions/{session_id}/events` — ordered Server-Sent Events progress stream.
- `GET /export/{session_id}?format=csv|json` — download session results.

## Complete script report

Example:

```json
{
  "original_text": "𐪀𐪁𐪂",
  "source_language": "ancient-north-arabian",
  "target_language": "en"
}
```

The returned report keeps these layers separate:

- exact original characters;
- script/language identity;
- historical variants;
- direction and direction description;
- approximate dating and dating status;
- geographic scope and materials;
- related scripts;
- Unicode blocks;
- transliteration systems;
- actual scholarly transliteration, if available;
- actual corpus/model translation, if available;
- confidence, provider and provenance.

## Voice

The web UI uses browser Speech Synthesis where available and provides Original / Transliteration / Translation plus Pause / Resume / Stop controls. Native pronunciation of an ancient language is not inferred from a modern TTS voice; the API reports `pronunciation_provider_required` when no scholarly pronunciation provider is installed.

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
