# Archaeological Knowledge & Ancient Language Intelligence Application

**Date:** 2026-09-12  
**Repository:** `amerhwitat/nlp`  
**Application:** `ArchaeologicalKnowledgeSystem/`

## Purpose

Create a self-contained archaeological knowledge and ancient-language research application inside `nlp`. It combines the existing Thamudic/Ancient North Arabian scanner capabilities with structured archaeological object records, provenance and location history, multilingual OCR/transliteration/translation, source-aware web research, image/IIIF support, OLTP/OLAP analytics, dashboards, and a source-grounded AI research assistant.

The application remains isolated from the repository's existing NLP, ThamudicScan, C++, .NET, Visual C++, and Python applications except through documented shared data contracts and reusable libraries.

## Domain model

Core entities: Object, Artifact, Souvenir, Inscription, Glyph, Script, Language, Site, Location, Excavation, ExcavationCampaign, HistoricalPeriod, DateAssertion, Material, ObjectType, Person/Entity, Museum, Collection, Authority, Loan, CustodyEvent, ProvenanceEvent, Publication/Source, Image/MediaAsset, IIIFManifest, OCRObservation, Transliteration, Translation, ScholarlyInterpretation, and Relationship.

Objects support exact or approximate dating, dating confidence, historical period, discovery location, excavation, present location, current status, ownership/custody, authority claims, related objects, inscriptions, images, and a complete historical location/provenance timeline. Status vocabulary includes discovered, excavated, stored, museum display, museum storage, loaned, returned, private collection, authority claim, disputed, missing, unknown, and transfer/repatriation states.

## Images and IIIF

Objects support front, back, left, right, top, bottom, detail, inscription, excavation, conservation, 3D/model preview, and custom views. Media records store source institution, source URL, creator, date, rights/license, attribution, checksum, dimensions, MIME type, and provenance. IIIF manifests/services are first-class resources for interoperable viewing, deep zoom, ordered views, and annotations.

The image harvester is rights-aware: it downloads only where automated reuse is permitted or explicitly supported by the source/API, otherwise retaining remote references and metadata. Copyright and license information is never discarded.

## Ancient-language/OCR architecture

The language subsystem is registry-driven. Initial adapters cover Old/Ancient North Arabian (Dadanitic, Safaitic, Hismaic, Taymanitic, Minaic, Thamudic variants), Old South Arabian, Egyptian hieroglyphic and related Egyptian writing, cuneiform workflows, Phoenician, Aramaic, historical Hebrew, Greek/Latin epigraphy, Nabataean, Palmyrene, Ugaritic, and additional Unicode-encoded ancient scripts as adapters become available.

The registry distinguishes script, language/dialect, Unicode availability, transliteration convention, OCR model availability, glyph segmentation, confidence, and translation capability. Unicode presence alone never implies reliable OCR or translation.

## Thamudic scanner integration

Integrate the existing scanner workflow as a reusable pipeline: image upload/drag-drop, live canvas, grayscale, CLAHE/contrast enhancement, adaptive thresholding, noise reduction, edge/shape extraction, inscription-region detection, glyph bounding boxes, manual correction, glyph classification, Unicode lookup, UTF-8 inspection, transliteration, RTL/LTR handling, boustrophedon orientation, Thamudic/Safaitic/Hismaic/Dadanitic/Early Arabic modes, confidence visualization, heatmaps/attention where supported, alternative glyph hypotheses, editable transcription, translation, and saving OCR observations against archaeological objects.

The existing canonical `data/ancient_north_arabian/alphabet.json` remains authoritative and is consumed through adapters rather than duplicated.

## Unicode and text

Canonical stored text is UTF-8/NFC, with NFD available for script analysis. Support code points, grapheme clusters, exact UTF-8 bytes, original-script strings, normalized transcriptions, transliterations, alternate readings, BCP-47 language tags, bidirectional text, RTL/LTR rendering, and historical orthographic variants. Egyptian hieroglyph property concepts such as sign sources, descriptions, functions, mirroring/rotation, and alternate sequences are represented through script-specific property adapters.

## Translation

Users select any registered modern destination language using BCP-47 identifiers. The pipeline is `image -> OCR -> glyph hypotheses -> transcription -> transliteration -> normalized ancient-language text -> translation -> target-language summary`.

Object, site, excavation, period, museum, and collection summaries use the same destination-language selector. Every translation stores source text/language, target language, engine/model, version, timestamp, confidence/quality indicators, terminology overrides, and source references.

Ancient languages without reliable machine-translation models use a clearly labeled research/reference mode combining lexicons, transliteration and LLM reasoning. Uncertainty is preserved rather than presented as fact.

The modern-language registry is not a hard-coded short list. It can represent all target languages needed by the deployment, including RTL/LTR languages, while translation providers may be local or remote. Provider credentials stay server-side.

## Knowledge graph and sources

The relational core is complemented by a relationship graph: object-site, object-excavation, object-inscription, inscription-script/language, object-period, object-object, ownership/custody, museum/collection, authority claim, source assertion, image depiction, OCR derivation, translation derivation, and scholarly interpretation.

CIDOC CRM-compatible mappings and external identifiers support cultural-heritage interoperability. Source records preserve external ID, URL, retrieval time, rights/license metadata, extraction method, and confidence.

Planned source adapters include Wikidata/Wikibase, Wikipedia where permitted, Europeana APIs, IIIF, Pleiades, Trismegistos, museum collection APIs, archaeological institutions, epigraphic databases, Unicode/script registries, and openly licensed excavation data/publications. Search results are evidence leads, not automatically authoritative facts.

## OLTP/OLAP

OLTP is normalized around stable entities and event histories. OLAP facts cover objects by period/region/site/material/type, excavation discoveries, inscriptions by script/language, museum and custody status, loans, dating confidence, OCR confidence, translation coverage, source coverage, and image coverage.

Generate SQL for PostgreSQL, MySQL, MariaDB, SQLite, SQL Server, Oracle-compatible deployments, and Microsoft Access where practical, with Access-compatible types and migration/import paths. JSON schemas and seeds provide portable/offline operation.

## AI research assistant

Combine retrieval, embeddings/entity similarity, deterministic archaeological metadata, OCR/transliteration results, and optional RNN/LLM models. Capabilities include object, inscription, excavation, site and period summaries; related-object discovery; same-feature/material searches; chronological and geographic comparison; source-conflict identification; multilingual summaries; and research question answering.

AI output must distinguish verified source fact, imported assertion, scholarly interpretation, OCR hypothesis, model inference, and unresolved uncertainty. Automated web research is rate-limited and provenance-preserving; automated writes require validation and audit events.

## Web and UI

Professional responsive views include research dashboard, object explorer/detail, provenance timeline, image/IIIF viewer, scanner/OCR workspace, transliteration editor, translation workspace, site/excavation explorer, map, chronology, knowledge graph, museum/collection dashboard, loans/claims dashboard, source browser, AI assistant, ingestion monitor, and data-quality dashboard.

Implement modular JavaScript/TypeScript, Web Components, PWA, Web Workers, WASM-compatible interfaces, Node.js, Deno, PHP standalone/API/database layers, HTML/CSS, shared JSON contracts, and tests.

## Security and testing

Server-side authorization is mandatory. APIs use validation, parameterized queries, CSRF protection for cookie-authenticated mutations, secure session cookies, security headers, rate limits, audit events, provenance tracking, and server-only external credentials. Tests cover Unicode/UTF-8, RTL/LTR, OCR preprocessing/segmentation, transliteration, translation contracts, provenance/location histories, database migrations, APIs, permissions, image rights, IIIF, workers/PWA, and deterministic AI/retrieval fixtures.

## Documentation

Provide README, architecture, archaeological ontology, data model, SQL/Access/JSON guides, language registry, Thamudic scanner, OCR/translation, multilingual translation, image/IIIF, ingestion/source adapters, AI/research assistant, security/provenance, deployment, API, and testing documentation.

## Acceptance criteria

1. Application isolated under `ArchaeologicalKnowledgeSystem/`.
2. Existing Thamudic/Ancient North Arabian data remains authoritative and reusable.
3. Objects can be dated, located, related to excavations/sites, assigned current status, and given full provenance/location history.
4. Multiple image views and rights metadata are supported.
5. Thamudic inscriptions can be scanned, segmented, corrected, transliterated, encoded, and translated.
6. Users can select any registered modern destination language for object information and summaries.
7. Ancient-script capability is registry- and confidence-aware.
8. SQL and JSON schemas exist for supported databases.
9. OLTP and OLAP models exist.
10. Dashboards expose archaeological and operational KPIs.
11. Web ingestion preserves source provenance.
12. AI output distinguishes evidence from inference.
13. Automated tests and deployment documentation are present.

## Standards informing the design

Unicode Old North Arabian and script data; Unicode Egyptian Hieroglyph Database (Unikemet); CIDOC CRM / ISO 21127:2023; IIIF Image and Presentation APIs; Pleiades; Trismegistos; Europeana APIs. These are interoperability references and not claims that this application owns or republishes their datasets.
