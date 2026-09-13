# Artifact database web integration

The web layer now exposes the Thamudic and general NLP scanner pipeline through a single artifact-oriented API while retaining the existing FastAPI scanner API.

## Architecture

`web upload/text -> media extraction -> source-language scan -> script scan -> transliteration -> evidence-backed translation -> ArtifactDatabase`

The Python modules remain the research implementation. JavaScript calls the HTTP API through `ThamudicScan/web_ui/src/artifacts.js`, so browser deployments do not duplicate ancient-language logic or silently diverge from the Python implementation.

## Entry point

```bash
uvicorn ThamudicScan.server.artifact_app:app --reload
```

The artifact entry point mounts the existing scanner application plus the artifact routers.

## Evidence preserved

Every analyzed artifact can retain original text, script/language identification, transliteration, translation status and confidence, media extraction metadata and SHA-256, source/provenance records, annotations and voice action metadata. Unsupported translations remain unavailable.

## Browser operations

`artifacts.js` provides calls for:

- text analysis
- ancient-language source-form analysis
- image/PDF/text uploads
- artifact search and retrieval
- database statistics
- JSON export

The API is intentionally JSON-first and can be consumed by React, plain browser JavaScript, other web clients, or future services.
