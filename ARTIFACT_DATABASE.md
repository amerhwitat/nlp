# Unified Thamudic / NLP Artifact Database

The repository now has a unified SQLite artifact evidence store at runtime (`data/artifacts.sqlite`). It combines the scanner evidence ledger with the earlier Ancient Object Research model.

## Stored layers

- Original artifact/inscription text and source/media type
- Source language and script variant
- Unicode/script scan results and codepoints
- Conservative transliteration
- Evidence-backed translation, target language, status, confidence, provider and provenance
- Historical object metadata: object type, culture, period, dates, material, technique, creator
- Geographic metadata: site, region, country, current location, latitude/longitude
- Bibliography, subjects, tags and competing readings
- Media extraction method, SHA-256 and extraction metadata
- Glyph/region annotations and reviewer candidates
- Voice/TTS actions as reproducible metadata
- Source/rights/license/record identifiers
- Arbitrary pipeline metadata

The store is evidence-first: an unsupported ancient translation remains `not_available` instead of being invented.

## Legacy database integration

`python/thamudic/artifact_legacy_adapter.py` migrates records from the existing `ancient_objects_db.py` ObjectDatabase into the unified ArtifactDatabase. The original database is never modified, and the complete legacy record is retained under artifact metadata.

## Python API

`python/thamudic/artifacts_db.py` provides `ArtifactDatabase` for creation, legacy migration, scan/translation/media/annotation/voice/provenance persistence, search, retrieval and statistics.

## Existing FastAPI artifact layer

The artifact-enabled FastAPI entry point reuses the scanner pipeline and exposes artifact analysis/search/persistence endpoints such as `/artifacts/stats`, `/artifacts/search`, `/artifacts/{artifact_id}`, `/artifacts/ingest`, `/artifacts/analyze`, `/artifacts/analyze-upload`, JSON export, annotation, provenance and voice capabilities.

Run:

```bash
uvicorn ThamudicScan.server.artifact_app:app --reload
```

## Node.js implementation

`nodejs/artifacts/` provides an independent Node.js implementation using Express + better-sqlite3:

- Persistent SQLite artifact catalog
- Historical object and inscription fields
- Search/filter API
- Ancient North Arabian Unicode scanner and conservative transliteration
- Media upload with SHA-256
- Annotation and provenance APIs
- Browser dashboard under `nodejs/artifacts/web/`

Run:

```bash
cd nodejs/artifacts
npm install
npm start
```

Then open `http://127.0.0.1:8090/`.

## Browser implementation

`nodejs/artifacts/web/` is a standalone HTML/CSS/JavaScript research surface. It provides dashboard statistics, artifact search, scanner/transliteration and evidence-policy presentation while consuming the Node API.

## External design research

The integration was informed by open-source ancient-artifact/epigraphy patterns such as spatial archaeological catalogs, structured inscription corpora, provenance-aware artifact records, linked ancient-world data and attestation-first ancient-language tooling. AncientMap combines archaeological-site catalogs, spatial data and a web API; openEtruscan combines corpus management, EpiDoc, prosopography and Linked Open Data; Hudhud combines Ancient South Arabian inscription search, detailed records, maps and semantic research; LAWD provides an ontology for connecting ancient-world artifacts, places, attestations and citations. These ideas are adapted as architectural patterns only; repository/data licenses must be checked before importing external datasets.

SQLite remains the canonical local storage format for the Python research implementation. Node.js provides a compatible web/API implementation rather than silently claiming binary database compatibility across arbitrary schema versions.
