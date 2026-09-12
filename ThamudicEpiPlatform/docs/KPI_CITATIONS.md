# KPI dashboard citations and definitions

## KPI design

The dashboard follows API-derived metrics rather than hardcoded UI counters. Core metrics include:

- historical objects and inscriptions
- readings and reviewed readings
- translation records and mean confidence
- PDF imports and exports
- processing errors
- processing latency when instrumentation is available
- per-language and per-script translation activity

## Standards

- [Unicode CLDR](https://cldr.unicode.org/) supplies internationalization and locale infrastructure.
- [Unicode BCP 47 Extensions](https://cldr.unicode.org/index/bcp47-extension) supplies machine-readable locale extension data.
- [Unicode Transliteration Guidelines](https://cldr.unicode.org/index/cldr-spec/transliteration-guidelines) establishes the distinction between transliteration and translation.
- [RFC 6497](https://www.rfc-editor.org/rfc/rfc6497) defines the BCP 47 transformed-content extension for transformations including transliteration, transcription and translation.

## Interpretation

Confidence is a model/reviewer signal, not a historical truth score. Aggregates must be accompanied by sample size and provenance when used for scholarly conclusions.
