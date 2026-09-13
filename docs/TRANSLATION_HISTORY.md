# Translation history, provenance, audit logging, and PDF exports

The NLP toolkit records every call through the universal ancient-language translation layer in an append-only JSON Lines log.

## Record contents

Each record contains:

- UTC timestamp
- original source text
- source language and source form
- target language and target form
- transliteration
- translation, when a provider actually supplies one
- status (`provider_required`, `direction_not_registered`, or provider-defined status)
- confidence
- provider name
- provenance/corpus/model information when supplied
- complete script-summary metadata
- non-secret request metadata
- SHA-256 record hash

The system deliberately does not store credentials, authorization headers, API keys, cookies, or arbitrary HTTP headers.

## Storage

Default path:

```text
translation_logs/translations.jsonl
```

Override with:

```text
THAMUDIC_TRANSLATION_LOG=/path/to/translations.jsonl
```

JSON Lines is used for durable append operations, streaming processing, and recovery of individual records.

## API

- `GET /translation-log` — return the log and hash verification summary.
- `GET /translation-log/export?format=json` — formatted JSON export.
- `GET /translation-log/export?format=jsonl` — JSON Lines export.
- `GET /translation-log/export?format=txt` — human-readable text export.
- `GET /translation-log/export?format=pdf` — paginated PDF report of the same records.
- `GET /export/{session_id}?format=pdf` — PDF report of persisted scanner results.
- `GET /script-summary/{language}/export?format=pdf` — PDF metadata report.
- `POST /script-report/export` with `{"format":"pdf"}` — PDF script/translation report.

PDF is an export/presentation format only; the canonical translation history remains the JSONL record set.

## PDF generation

PDF generation is implemented in `python/thamudic/pdf_export.py` using ReportLab Platypus. Platypus separates document layout from content and supports flowable paragraphs, tables and multi-page documents. urlReportLab Platypus documentationhttps://docs.reportlab.com/reportlab/userguide/ch5_platypus/

The dependency is `reportlab>=5.0`. ReportLab 5.0 was released in June 2026 and retained the PDF-generation behavior of the previous release line while applying security-related settings changes. citeturn0search7turn0search5

If ReportLab is unavailable, the application remains importable and the PDF endpoint reports a clear dependency error. This prevents optional PDF support from breaking the scanner itself.

## Integrity

Every record has a deterministic SHA-256 hash over its canonical JSON representation, excluding the hash field itself. `verify_records()` recalculates hashes and reports invalid entries.

This is an integrity aid, not a cryptographic signature or tamper-proof ledger. For stronger provenance, retain the exported file together with external signatures, immutable storage, or a versioned research archive.

## Research provenance

OCIANA provides readings in roman transliteration, English translations, references, commentary, bibliography, provenance, relationships to other texts, and images/facsimiles. The logging layer is designed to preserve these kinds of provenance fields when a translation provider supplies them. urlOCIANAhttps://ociana.osu.edu/

ORACC's ATF ecosystem distinguishes language/dialect metadata and Unicode transliteration conventions; its JSON text editions include transliteration and lemmatization and are structured hierarchically at text/surface/column/line/word/sign levels. urlORACC ATF Primerhttps://build-oracc.museum.upenn.edu/doc/help/editinginatf/primer/index.html

## Voice and logging

Voice playback is separate from scholarly pronunciation. Browser speech synthesis can speak supported modern/transliterated text, while native ancient pronunciation remains provider-dependent. Browser recognition support is also browser-dependent; Web Speech exposes both `SpeechSynthesis` and `SpeechRecognition`, and recognition may use an online service unless on-device processing is explicitly supported and enabled. urlMDN Web Speech APIhttps://developer.mozilla.org/en-US/docs/Web/API/Web_Speech_API

## Safety and scholarship

A registered translation direction is a capability declaration, not proof that a model or parallel corpus exists. When no attested provider is available, the application records the attempt with `provider_required` instead of inventing a translation. This distinction is essential for ancient-language research.
