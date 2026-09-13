# Unified Thamudic / NLP Artifact Database

The repository now has a dedicated SQLite artifact evidence store at runtime (`data/artifacts.sqlite`). It is designed to preserve the complete research pipeline rather than only a final translation.

## Stored layers

- Original artifact text and source/media type
- Source language and script variant
- Unicode/script scan results and codepoints
- Conservative transliteration
- Evidence-backed translation, target language, status, confidence, provider and provenance
- Media extraction method, SHA-256 and extraction metadata
- Glyph/region annotations and reviewer candidates
- Voice/TTS actions as reproducible metadata (not audio blobs)
- Source/rights/license/record identifiers
- Arbitrary pipeline metadata and tags

The store is evidence-first: an unsupported ancient translation remains `not_available` instead of being invented.

## Python API

`python/thamudic/artifacts_db.py` provides `ArtifactDatabase` for creation, scan/translation/media/annotation/voice/provenance persistence, search, retrieval and statistics.

## Web API

The artifact-enabled entry point reuses the existing FastAPI scanner and adds:

- `GET /artifacts/stats`
- `GET /artifacts/search?q=&source_language=&script_variant=&limit=`
- `GET /artifacts/{artifact_id}`
- `POST /artifacts/ingest`
- `POST /artifacts/analyze`
- `POST /artifacts/analyze-ancient`
- `POST /artifacts/analyze-upload`
- `GET /artifacts/export/json`
- `POST /artifacts/{artifact_id}/annotation`
- `POST /artifacts/{artifact_id}/provenance`
- `POST /artifacts/{artifact_id}/voice`
- `GET /artifacts/voice/capabilities`
- `GET /artifacts/script/{language}`

Run the artifact-enabled server with:

```bash
uvicorn ThamudicScan.server.artifact_app:app --reload
```

## Web client

`ThamudicScan/web_ui/src/artifacts.js` exposes browser-side functions for analysis, ancient-language analysis, uploads, artifact search/retrieval/statistics and JSON export. The existing scanner UI remains compatible; deployments that use the artifact entry point can call these APIs without duplicating the Python pipeline in JavaScript.

## Database policy

SQLite remains the canonical local storage format. The artifact database is separate from the scanner session database so operational scan progress and long-lived scholarly evidence are not conflated. Both are portable and can be backed up as ordinary SQLite files.
