# Ancient Language Intelligence Expansion Specification

**Status:** Approved design; implementation pending execution-mode selection.

## Goal

Expand ArchaeologicalKnowledgeSystem so it can identify, scan, encode, transliterate, reason about, and persist evidence for ancient Mesopotamian, Egyptian, Levantine, Ancient North/South Arabian, Greek, Latin, and related ancient writing systems across historical periods.

## Scope

The system will support a registry-driven architecture rather than a hard-coded language list. Initial registry coverage includes:

- Mesopotamia and neighboring cuneiform traditions: Sumerian, Akkadian, Babylonian, Assyrian, Elamite, Eblaite, Hurrian, Urartian, Hittite and related cuneiform corpora.
- Egypt: Egyptian hieroglyphs, Old/Middle/New Egyptian, Late Egyptian, Demotic and related Egyptian writing stages.
- Levant/Northwest Semitic: Aramaic, Imperial Aramaic, Phoenician, Ugaritic, Hebrew and related scripts.
- Ancient Arabia: Safaitic, Hismaic, Thamudic, Dadanitic, Taymanitic, Nabataean, Old North Arabian and Old South Arabian.
- Classical Mediterranean: Ancient/Classical Greek and Latin, with room for Old Italic and related scripts.
- Unicode-encoded neighboring ancient scripts are registry-extensible without changing application code.

Unicode encoding is treated as an evidence/representation capability, not as proof that OCR, language identification, transliteration, morphology, or translation is available.

## Recognition pipeline

```text
Image/PDF/Object
 -> document/page analysis
 -> script-family detection
 -> historical-period estimation
 -> language/dialect detection
 -> direction/layout detection
 -> text-region detection
 -> glyph/sign segmentation
 -> glyph/sign recognition
 -> Unicode/code-point resolution
 -> original reconstructed text
 -> scholarly transliteration
 -> morphology/lexicon analysis
 -> semantic/context reasoning
 -> translation
 -> confidence + competing readings
 -> provenance + database + knowledge graph
```

The engine must preserve alternate hypotheses instead of collapsing uncertain readings into one answer.

## Registry model

Every language/script record may contain:

- stable internal identifier
- language family
- language/stage/dialect
- script and writing-system type
- geographic range
- historical periods and date ranges
- direction and layout rules
- Unicode blocks/code-point ranges
- UTF-8 representation
- aliases and scholarly names
- transliteration conventions
- glyph/sign identifiers
- variant forms
- rotation/mirroring/combining behavior
- external authority identifiers
- OCR model and version
- segmentation model
- language identification model
- morphology/lexicon resources
- translation capability
- semantic reasoning capability
- evidence sources and licenses
- capability level
- confidence and validation status

Capability levels:

`ENCODED`, `REFERENCE`, `SEGMENTABLE`, `OCR_READY`, `LANGUAGE_DETECTABLE`, `TRANSLITERATION_READY`, `MORPHOLOGY_READY`, `TRANSLATION_READY`, `SEMANTIC_REASONING_READY`.

## Unicode and UTF-8

The registry will use Unicode as the normative character-encoding layer and store code point, block, Unicode name, UTF-8 bytes, normalization form, aliases, and script-specific identifiers where available. The implementation will preserve source text exactly where possible and retain normalized forms separately.

Egyptian hieroglyphs must integrate Unicode's Unikemet properties, including Gardiner/Unikemet identifiers, core status, source/catalog identifiers, rotation and mirroring properties, and extended repertoire metadata. Unikemet data is UTF-8/NFC and must be imported with source/version/provenance metadata rather than copied as undocumented application constants.

## Neural architecture

Use a hierarchical recognition system:

```text
visual encoder
 -> script classifier
 -> script-specific recognizer
 -> language/period classifier
 -> Unicode resolver
 -> transliteration engine
 -> morphology/lexicon engine
 -> contextual reasoning engine
 -> translation engine
```

Script-specific adapters allow cuneiform, hieroglyphic, alphabetic, abjad, logosyllabic and other writing systems to use different segmentation and recognition strategies.

Models are versioned. Each prediction records model identifier/version, input asset, preprocessing configuration, candidate list, confidence, timestamp, and provenance.

## Cuneiform requirements

Cuneiform records must distinguish visual sign identity from linguistic value. A sign occurrence may have multiple readings depending on language, period, context, determinative/function and lexical environment. The data model therefore stores sign, sign variant, sign value, reading, language, period, context and hypothesis separately.

## Egyptian requirements

Egyptian hieroglyph processing must support Unicode hieroglyphs, Gardiner-style identifiers, Unikemet metadata, rotation/mirroring rules, transliteration, word/lemma lookup, period/date assertions and uncertainty. Where Unicode cannot represent a source form, an external encoding such as Manuel de Codage may be retained as a source representation without replacing Unicode.

## Database ER model

Core entities:

```text
CULTURE 1---N PERIOD
PERIOD 1---N LANGUAGE_STAGE
LANGUAGE 1---N LANGUAGE_STAGE
LANGUAGE_STAGE N---N SCRIPT
SCRIPT 1---N GLYPH_SIGN
GLYPH_SIGN 1---N UNICODE_CHARACTER

SITE 1---N EXCAVATION
EXCAVATION 1---N OBJECT
OBJECT 1---N INSCRIPTION
INSCRIPTION 1---N OCR_OBSERVATION
OCR_OBSERVATION 1---N GLYPH_OCCURRENCE
INSCRIPTION 1---N TRANSLITERATION
INSCRIPTION 1---N TRANSLATION
INSCRIPTION 1---N INTERPRETATION

LANGUAGE 1---N LEXEME
LEXEME 1---N LEXICAL_SENSE
LEXEME 1---N MORPHEME

OBJECT 1---N MEDIA_ASSET
OBJECT 1---N PROVENANCE_EVENT
ENTITY 1---N ASSERTION
SOURCE 1---N ASSERTION
ASSERTION N---1 ENTITY
```

All relationships must support provenance and confidence where a relationship is observational, imported, scholarly, inferred or unresolved.

## MADM model

The Multidimensional Ancient-language Data Model is the analytical/star-schema layer over normalized OLTP data. Dimensions include:

- Time/period
- Geography/site
- Culture
- Language/stage/dialect
- Script/writing system
- Glyph/sign
- Unicode
- Object/artifact
- Material/object type
- Text/inscription
- Source/publication
- OCR/model
- Translation engine/model
- Confidence/evidence class

Facts include glyph recognition, inscription observations, language classification, transliteration, lexical analysis, translation, provenance, dating and source assertions.

The MADM must support cross-dimensional queries such as comparing readings of a sign across periods, languages and archaeological sites, and measuring OCR/translation coverage and confidence.

## Evidence and provenance

The application distinguishes:

- `verified_unicode_fact`
- `archaeological_source`
- `scholarly_assertion`
- `ocr_observation`
- `neural_prediction`
- `lexical_candidate`
- `translation_hypothesis`
- `semantic_inference`
- `uncertain`

No model-generated interpretation is silently promoted to an authoritative historical fact.

## Web knowledge ingestion

Web ingestion is source-aware and provenance-preserving. Normative Unicode material is prioritized for encoding metadata. Scholarly/curated corpora are stored as source assertions and external references. Initial research sources include Unicode, ORACC/ETCSRI, CDLI and the Thesaurus Linguae Aegyptiae, with additional openly accessible scholarly corpora added through adapters.

The crawler stores source URL/identifier, retrieval timestamp, license/rights where available, extraction method, source version, checksum when applicable, and validation state. Duplicate facts are reconciled by authority and provenance rather than by blind overwrite.

## Translation output

For every translated inscription attached to an object, the UI/API returns:

1. Original source text/glyph representation.
2. Scholarly transliteration.
3. English-readable transliteration.
4. Arabic-readable transliteration.
5. User-selected target-language translation.
6. Morphological/lexical reasoning where available.
7. Alternative readings and translations.
8. Evidence and provenance.
9. Confidence and model/source versions.

## Security and rights

Remote model credentials remain server-side. PDF/image ingestion is bounded and validated. External content is stored with provenance and rights metadata. The system does not imply ownership of third-party images or corpora merely because metadata was ingested.

## Validation

The implementation must include tests for:

- Unicode and UTF-8 round trips.
- Registry consistency and historical-period links.
- Script/language capability declarations.
- Cuneiform multi-reading representation.
- Egyptian Unikemet identifiers/properties.
- OCR hypothesis provenance.
- Transliteration and translation provenance.
- ER schema integrity.
- MADM dimensional/fact integrity.
- Web-source ingestion and deduplication.
- End-to-end image/PDF -> recognition -> Unicode -> transliteration -> translation -> persistence.

## Research basis

The design uses current Unicode script data and the Unicode Egyptian Hieroglyph Database (Unikemet), which documents Egyptian Hieroglyphs and Egyptian Hieroglyphs Extended-A plus sign properties and source identifiers. Unicode's supported-script registry also identifies Sumero-Akkadian Cuneiform, Old Persian Cuneiform, Egyptian Hieroglyphs, Imperial Aramaic, Old South Arabian, Nabataean, Old North Arabian, Phoenician, Ugaritic, Greek and Latin among encoded scripts. ORACC provides curated, richly annotated cuneiform corpora, including Sumerian royal inscriptions, while the Thesaurus Linguae Aegyptiae supports Unicode hieroglyph search, transliteration and date/attestation filtering.

References:
- Unicode Standard Annex #57, Unicode Egyptian Hieroglyph Database (Unikemet), Unicode 18.0.0.
- Unicode Supported Scripts.
- Unicode Standard, Chapter 6, Writing Systems and Punctuation.
- ORACC / Electronic Text Corpus of Sumerian Royal Inscriptions.
- Thesaurus Linguae Aegyptiae.

These references are evidence sources for registry design; their content is not treated as a substitute for scholarly validation.
