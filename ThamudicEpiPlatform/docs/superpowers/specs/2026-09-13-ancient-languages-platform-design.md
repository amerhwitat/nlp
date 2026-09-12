# Ancient Languages Cross-Script NLP Platform — Design Specification

**Date:** 2026-09-13  
**Status:** Approved architecture; implementation follows after written-spec review  
**Repository:** `amerhwitat/nlp`  
**Primary application:** `ThamudicEpiPlatform/`

## 1. Purpose

Expand `ThamudicEpiPlatform` from an Ancient North Arabian/Thamudic research integration layer into a provenance-aware, Unicode-native ancient-language research platform covering the requested Mesopotamian, Egyptian, Arabian, Greek, and Latin domains while preserving the existing Thamudic implementation as a canonical component.

The implementation will combine reusable code and data contracts already present elsewhere in `amerhwitat/nlp` with the platform's FastAPI/SQLite/React architecture. Shared functionality will be promoted into language-neutral registries and service interfaces rather than duplicated per language.

The system will support:

- Unicode code points, normalization, UTF-8 encoding and exact byte representations.
- Script/language identification and historical-stage metadata.
- Scholarly transliteration plus English and Arabic transliteration/rendering.
- Translation targets represented by BCP 47/CLDR language identifiers rather than a hardcoded finite language list.
- Literal, scholarly/interlinear and meaning-oriented translation modes.
- Proofing, morphology, disambiguation and confidence scoring through a hybrid neural + symbolic pipeline.
- Source/provenance preservation and explicit uncertainty/alternative readings.
- Speech/pronunciation with modern TTS and reconstructed/approximate ancient pronunciation profiles.
- Cross-language client libraries and automated Windows CMD, PowerShell, Bash and container builds/deployments.

## 2. Scope

### Core language/script families

**Mesopotamia and adjacent cuneiform:** Sumerian; Akkadian; Assyrian and Babylonian historical varieties; Elamite; Eblaite metadata where evidence supports it; Ugaritic as a Levantine cuneiform language.

**Egypt:** Ancient Egyptian historical stages; hieroglyphic and hieratic metadata; Demotic; Coptic.

**Arabia and neighboring Semitic epigraphy:** Ancient North Arabian; Safaitic; Hismaic; Dadanitic; Old South Arabian including Sabaic, Minaic, Qatabanian and Hadramitic; Old/Ancient Arabic; Nabataean.

**Greek:** Mycenaean/Linear B; Archaic Greek; Classical Greek; Hellenistic/Koine Greek; later ancient/Byzantine metadata where useful for historical continuity.

**Latin:** Old Latin; Classical Latin; Late Latin, with extensibility for Medieval/Neo-Latin without forcing those stages into the ancient-language models.

**Adjacent registry:** Hebrew, Phoenician, Aramaic, Imperial Aramaic and Syriac may be represented in the common registry where they share script/transliteration/epigraphic infrastructure, but are not allowed to displace the five requested core domains.

### Modern target languages

Modern target languages are represented through a provider-neutral registry keyed by BCP 47/CLDR identifiers. The platform must not claim that every target has equal translation or speech quality. A target is enabled only when an installed provider/model/translation resource supports it; otherwise the UI reports the capability as unavailable instead of fabricating output.

## 3. Architectural principles

1. **Unicode first:** Unicode scalar values and normalized text are canonical; UTF-8 is the interchange/storage encoding.
2. **Transliteration is not translation:** script conversion, scholarly romanization, pronunciation guidance and semantic translation remain distinct operations.
3. **Evidence before inference:** source text, glyph observations, scholarly readings, dictionary evidence, model hypotheses and human corrections are stored separately.
4. **Uncertainty is data:** alternate readings, damaged signs, lacunae, uncertain morphology and model confidence are first-class records.
5. **No fabricated ancient-language knowledge:** unsupported language/model combinations return an explicit unsupported state.
6. **Hybrid NLP:** deterministic Unicode/transliteration/morphology rules are combined with statistical/neural components and lexical resources.
7. **Provider-neutral interfaces:** translation and speech engines are adapters; no provider is hardcoded into the core database contract.
8. **License/provenance aware:** public projects may inform architecture, but source code is not copied without compatible licensing. Every imported dataset or algorithm receives provenance metadata.
9. **Backward compatibility:** existing Thamudic APIs, tables, clients and canonical mappings remain usable.
10. **Cross-language parity:** Python is the reference implementation; C++, C#, Java, Go, Rust, JavaScript and TypeScript expose compatible contracts where practical.

## 4. Canonical data model

Add a versioned language/script registry under `languages/registry/` and `data/languages/` with JSON/CSV fixtures suitable for import into SQLite/PostgreSQL.

Each language record contains:

- stable internal identifier;
- ISO/BCP 47/CLDR identifiers when available;
- language family and historical stage;
- script identifiers;
- directionality;
- Unicode ranges/blocks and relevant combining marks;
- normalization policy;
- transliteration systems and aliases;
- pronunciation/TTS capability;
- evidence/status notes;
- source citations and license/provenance.

Each script record contains:

- Unicode script name/code;
- Unicode version observed;
- code-point ranges;
- character names and aliases where available;
- shaping/layout/directionality metadata;
- normalization requirements;
- transliteration mappings;
- historical usage notes.

### Database additions

Create a migration rather than replacing the existing schema. Initial tables:

- `languages`
- `scripts`
- `language_scripts`
- `script_ranges`
- `transliteration_systems`
- `transliteration_rules`
- `translation_targets`
- `lexemes`
- `morphemes`
- `proofing_predictions`
- `translation_results`
- `translation_alternatives`
- `neural_predictions`
- `speech_profiles`
- `speech_jobs`
- `model_registry`
- `evaluation_runs`
- `provenance_records`

Existing `sources`, `periods`, `objects`, `annotations`, `readings`, `scan_sessions` and `asset_manifest` remain authoritative for current platform records.

## 5. Processing pipeline

```text
Image / inscription / Unicode text
              |
       language + script ID
              |
     Unicode normalization
              |
   glyph/token segmentation
              |
 OCR/glyph candidates (if image)
              |
 scholarly transliteration candidates
              |
 +-------------+------------------+
 |             |                  |
Arabic        English        target-script
rendering     rendering       transliteration
 |             |                  |
 +-------------+------------------+
              |
 morphology + lexicon + context
              |
 literal / interlinear / scholarly
              |
 semantic / meaning translation
              |
 proofing + confidence + alternatives
              |
 speech/IPA/pronunciation profile
              |
 provenance + export
```

Each stage has a stable request/response schema so stages can be tested independently and reused by all language clients.

## 6. Transliteration and UTF-8

The transliteration engine will support multiple named systems per language. Rules are directional and optionally reversible. Output types are explicitly distinguished:

- `unicode_native`
- `utf8`
- `scholarly_transliteration`
- `latin_transliteration`
- `arabic_transliteration`
- `target_script`
- `ipa`

UTF-8 output includes exact bytes when requested. The system records normalization form (`NFC`, `NFD`, `NFKC`, `NFKD`) and never silently changes normalization in a way that would invalidate scholarly comparison.

Egyptian hieroglyph data will additionally support quadrat/layout metadata and Unikemet/Gardiner-related identifiers rather than treating hieroglyphs as ordinary linear letters.

## 7. Translation and proofing

Translation modes:

1. **Literal:** closest defensible lexical/morphological rendering.
2. **Scholarly:** reading plus evidence and notes, preserving uncertainty.
3. **Interlinear:** token, transliteration, morphology and gloss alignment.
4. **Meaning:** context-sensitive natural-language rendering.
5. **Proofing:** spelling/glyph/morphology/readability suggestions without silently altering source data.

The neural layer will provide:

- language/script classification;
- glyph/token candidate ranking;
- transliteration correction suggestions;
- morphological tagging;
- lexical retrieval/ranking;
- contextual disambiguation;
- translation hypotheses;
- confidence calibration;
- alternative hypotheses;
- human-review feedback storage.

The first implementation should be CPU-safe and modular. Small deterministic/statistical baselines and adapter interfaces are preferred over committing large model binaries to Git. Optional model packages may be loaded from configured local/model registries.

## 8. Modern-language target layer

The target registry will consume CLDR/BCP 47-style identifiers and provider capability metadata. It must support at minimum the modern languages offered by installed translation providers and Unicode/CLDR locale data, with no artificial fixed-language ceiling.

A target capability record contains:

- BCP 47 language tag;
- script/region where applicable;
- translation direction;
- transliteration capability;
- TTS capability;
- provider/model identifier;
- quality/evaluation status;
- fallback chain.

## 9. Speech and pronunciation

Speech is an adapter layer with controls for:

- voice;
- language/locale;
- historical pronunciation profile;
- speed/rate;
- pitch when supported;
- volume;
- pause/play/resume/stop;
- token/line playback;
- IPA/phoneme display;
- source/transliteration/translation playback;
- audio export when the provider supports it.

Ancient-language speech must be labeled as reconstructed, scholarly, approximate or provider-derived. The application must not represent reconstructed pronunciation as an objectively known recording of an ancient speaker.

## 10. API

Add versioned endpoints while retaining existing endpoints:

- `GET /api/languages`
- `GET /api/scripts`
- `GET /api/languages/{id}`
- `GET /api/targets`
- `GET /api/models`
- `POST /api/normalize`
- `POST /api/transliterate`
- `POST /api/proof`
- `POST /api/translate`
- `POST /api/translate/interlinear`
- `POST /api/speech/profile`
- `POST /api/speech/synthesize`
- `GET /api/speech/jobs/{id}`
- `GET /api/provenance/{id}`

Requests and responses must carry language/script IDs, model/provider IDs, normalization information, confidence and provenance where applicable.

## 11. Web UI

Extend the existing React/TypeScript UI with:

- source language/script selector;
- historical-stage selector;
- Unicode/UTF-8 inspector;
- native-script editor;
- transliteration panels for scholarly, English and Arabic views;
- modern target-language selector;
- literal/scholarly/interlinear/meaning tabs;
- proofing and confidence panel;
- alternative-reading panel;
- provenance/source panel;
- speech voice/profile controls;
- token/line playback;
- export of Unicode, UTF-8 bytes, transliteration, translation and metadata.

Accessibility, RTL layout and keyboard navigation are mandatory for Arabic and bidirectional material.

## 12. Cross-language implementations

Promote shared contracts into language-neutral schemas and implement clients/adapters in:

- Python;
- C++;
- C#/.NET;
- Java;
- Go;
- Rust;
- JavaScript;
- TypeScript/web.

Existing `nlp` language-specific implementations and registries are reused through documented imports/generation rather than manually copied tables. Build metadata is added where missing so each implementation can build independently.

## 13. Automation

Update and unify:

- `scripts/*.sh`
- `scripts/*.ps1`
- `scripts/*.bat`
- Docker/Compose files
- GitHub Actions

Automation must provide explicit commands for:

1. dependency installation;
2. registry generation/validation;
3. database migration;
4. Python tests;
5. C++ build/tests;
6. C# build/tests;
7. Java build/tests;
8. Go build/tests;
9. Rust build/tests;
10. JS/TS lint/test/build;
11. web production build;
12. server launch;
13. container build/run;
14. clean/rebuild;
15. deployment packaging.

PowerShell scripts must remain compatible with Windows PowerShell 5.1 where feasible; PowerShell 7-only syntax must not be required for the base build path.

## 14. Testing and evaluation

Add tests for:

- Unicode scalar/code-point correctness;
- UTF-8 byte round trips;
- NFC/NFD behavior;
- script/language registry integrity;
- transliteration round trips where reversible;
- Arabic RTL rendering data;
- damaged/uncertain readings;
- proofing confidence bounds;
- translation provider capability negotiation;
- unsupported-language fail-closed behavior;
- speech profile validation;
- API contracts;
- database migrations;
- cross-language client parity;
- build scripts on Windows and POSIX paths.

Evaluation datasets must distinguish gold-standard scholarly data from model-generated outputs. Metrics and human-review results are stored with dataset/model provenance.

## 15. Research and implementation inspirations

The implementation may draw architectural lessons from Unicode/CLDR, READ-style ancient-document systems, Potnia, CuReD and browser-based ancient-language OCR projects. It must not copy incompatible/proprietary source code. External code may only be incorporated when its license permits redistribution and the repository records the license and source location.

## 16. Documentation and citations

Update:

- root `README.md`;
- `ThamudicEpiPlatform/README.md`;
- `docs/SOURCES.md`;
- API documentation;
- architecture documentation;
- language-specific READMEs;
- deployment/build documentation;
- model/provenance documentation;
- cross-language implementation READMEs.

Every language implementation README must cite its registry source, Unicode/CLDR references, external algorithm/data sources and applicable licenses. Documentation must distinguish Unicode encoding facts from linguistic/translation claims.

## 17. Security and provenance

No automatic crawling of unrelated sites or dark-web resources is introduced by this specification. Any source ingestion remains authorization- and robots-aware. Uploaded research data is isolated from executable model payloads. Provider credentials remain outside source control. Translation/TTS adapters receive only the minimum required text and metadata.

## 18. Acceptance criteria

The implementation is complete when:

- the requested ancient-language/script registry is queryable;
- Unicode/UTF-8 normalization and byte tests pass;
- Thamudic/Ancient North Arabian functionality remains backward compatible;
- transliteration can produce scholarly, English and Arabic forms where a mapping exists;
- target-language capabilities are discoverable dynamically;
- literal/interlinear/meaning translation paths expose provenance and uncertainty;
- proofing produces confidence-bearing suggestions rather than destructive edits;
- speech controls work for supported modern voices and correctly label reconstructed ancient pronunciation;
- all maintained language clients build from their own subdirectories;
- CMD, PowerShell, Bash and container automation cover install/build/test/deploy;
- documentation contains source/license citations;
- CI exercises the core API, registry, web build and representative language clients.

## 19. Primary references

- Unicode Standard and supported scripts: https://www.unicode.org/standard/supported.html
- Unicode 17.0 Core Specification: https://www.unicode.org/versions/Unicode17.0.0/
- Unicode Egyptian Hieroglyph Database (Unikemet): https://www.unicode.org/reports/tr57/
- Unicode CLDR: https://cldr.unicode.org/
- CLDR language identifiers: https://cldr.unicode.org/index/cldr-spec/picking-the-right-language-code
- CLDR transliteration guidelines: https://cldr.unicode.org/index/cldr-spec/transliteration-guidelines
- Existing repository source index: `README.md`
- Existing platform source/licensing notes: `ThamudicEpiPlatform/README.md`
- Existing platform schema/provenance model: `ThamudicEpiPlatform/database/schema.sql`

## 20. Implementation sequence

1. Promote and validate existing `nlp` registries/contracts.
2. Add language/script registry and Unicode validation layer.
3. Add transliteration engines and test vectors.
4. Add database migration and provenance/evidence tables.
5. Add proofing/morphology/disambiguation interfaces and CPU baseline models.
6. Add translation adapters and target-language capability registry.
7. Add speech profiles/adapters and web controls.
8. Add cross-language clients/build metadata.
9. Update documentation/citations and automation.
10. Run verification matrix and open the implementation PR.

This specification deliberately separates the design decision from implementation so that the large multi-language expansion remains auditable, reversible and testable.