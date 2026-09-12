# Sources and code citations

## Supplied deployments

1. `https://thamudicscan-s3wz30.public.builtwithrocket.new/` — supplied public Thamudic Scanner deployment.
2. `https://chimera-ii-os-730893.onhercules.app/` — supplied public Chimera II OS deployment.
3. `https://thamudic-scanner.softr.app/` — supplied Softr Thamudic Scanner deployment.

These URLs are integration references. This execution environment returned cache misses for all three, so no claim is made that their private source code or database internals were recovered. Run `tools/site_audit.py` from an authorized network to produce an asset manifest.

## Open-source research references

- READ — Research Environment for Ancient Documents: https://github.com/readsoftware/read — ancient-document research environment with linked images, parallel transcriptions, translations, glossaries and paleographic charts; GPLv3. citeturn0search3
- Cuneiform Detector: https://github.com/marie-saccucci/cuneiform-detector — open-source FastAPI + React inscription-image detection architecture. citeturn0search2
- Textorcist: https://github.com/bhagesh-h/textorcist — React/TypeScript/Vite client-side OCR UI patterns, including local processing and export. citeturn0search4
- OpenEtruscan: https://www.openetruscan.com/docs — open computational epigraphy platform with corpus, API and scholarly provenance/evaluation concepts. citeturn0search10

## Public Thamudic image references

Wikimedia Commons contains Thamudic inscription imagery under explicit Creative Commons licenses, for example a Qaryat al-Faw fragment under CC BY 4.0 and a Jubbah inscription under CC BY 2.0. These are suitable examples for provenance-aware dataset ingestion when the license requirements are retained. citeturn1search1turn1search2

## Existing repository sources

- `python/thamudic/old_north_arabian.py` — canonical Unicode registry/transliteration.
- `ThamudicScan/server/` — existing FastAPI scanner backend.
- `ThamudicScan/web_ui/` — existing React scanner UI.
- `SOFTR_DATABASE_MIGRATION.md` — repository's previously documented Softr migration model.
- `THAMUDIC_SCANNER_RESEARCH.md` — project research notes.

## Citation policy

Every imported external source should retain URL, retrieval timestamp, license/rights statement, content hash and provenance. Do not redistribute a third-party minified JavaScript bundle merely because it is publicly reachable; reproduce behavior with clean-room code or import it only when its license explicitly permits redistribution.
