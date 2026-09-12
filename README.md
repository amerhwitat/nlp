# NLP / Ancient Language Research Toolkit

This repository contains Python, C++, .NET, Visual C++ and desktop/web implementations for Thamudic, Ancient North Arabian and the expanding ArchaeologicalKnowledgeSystem ancient-language research platform.

## Ancient-language intelligence expansion

The registry-driven architecture is being extended across Mesopotamian cuneiform, Egyptian writing, Ancient North/South Arabian, Coptic, Greek, Latin, Ancient/Modern Aramaic, Syriac, Mandaic and Ancient/Modern Hebrew.

### Hebrew

Hebrew historical stages are represented separately:

- Ancient/Biblical Hebrew
- Second Temple/historical Hebrew
- Masoretic Hebrew
- Medieval Hebrew
- Modern Hebrew
- Paleo-Hebrew historical script context
- Samaritan Hebrew related/distinct tradition

Hebrew registry data preserves RTL direction, final forms, niqqud, cantillation and combining behavior.

## Unicode, UTF-8 and alphabet/sign registry

The application ships structured alphabet/sign tables and a versioned offline Unicode snapshot. Each character/sign record can retain:

- language and historical stage
- script/script variant
- Unicode code point
- Unicode name
- exact UTF-8 bytes
- normalization and bidi metadata
- transliteration
- pronunciation profile
- provenance/source/version/checksum

`tools/generate_ancient_unicode_registry.py` rebuilds the Unicode snapshot from a pinned Unicode Character Database. This is intentionally separate from language identity: Unicode Script does not by itself identify a language.

For cuneiform and Egyptian hieroglyphs, the registry models signs and values rather than pretending that every writing system is an alphabet.

## Pronunciation and voice

Transliteration and target-language translation fields now have browser speech controls for:

- Listen
- Pause
- Resume
- Stop
- Replay
- voice selection
- locale
- rate
- pitch
- volume
- pronunciation mode

Modern speech, scholarly pronunciation, reconstructed pronunciation and reference pronunciation are distinct. Ancient reconstructed speech is explicitly labeled and is never represented as an authenticated historical recording without appropriate evidence.

## ThamudicScan web application

Public deployment:

https://thamudicscan-s3wz30.public.builtwithrocket.new/

The hosted application provides a browser-accessible interface for the Thamudic / North Arabian research tooling maintained in this repository.

## Ancient North Arabian Unicode support

The repository includes a canonical registry at `data/ancient_north_arabian/alphabet.json` covering the complete Unicode Old North Arabian block `U+10A80–U+10A9F`:

- 29 letters
- 3 encoded numbers
- Unicode character
- Unicode code point
- scholarly transliteration
- exact UTF-8 byte sequence
- Dadanitic encoding basis
- variant-script metadata for Safaitic, Hismaic, Taymanitic, Minaic and Thamudic B

Unicode encodes Old North Arabian using Dadanitic forms. Variant historical forms are represented as variant/font metadata rather than fabricated Unicode code points.

## Implementations

- `cpp/thamudic/` — C++20 library and Unicode registry.
- `vcpp/` — Visual Studio native Windows desktop scanner.
- `dotnet/` — CLI, WPF desktop and web implementations.
- `python/thamudic/` — Python Unicode/UTF-8/transliteration API.
- `python/ancient_languages/` — shared historical-language, Unicode and pronunciation services.
- `python/tests/` — Unicode, Hebrew and pronunciation regression tests.
- `data/ancient_languages/` — language-stage, alphabet, Unicode and pronunciation registries.
- `db/sql/` — normalized ER and analytical/MADM schemas.
- `web/javascript/` — browser workbench and voice controls.
- `ThamudicScan/` — documentation and link to the public web deployment.

## Documentation

- `docs/superpowers/specs/2026-09-12-ancient-language-intelligence-expansion.md`
- `docs/superpowers/specs/2026-09-12-hebrew-alphabet-utf8-voice-addendum.md`
- `docs/superpowers/plans/2026-09-12-hebrew-alphabet-utf8-voice-plan.md`
- `docs/ANCIENT_LANGUAGE_VOICE.md`

## Sources

Unicode's Character Database and code charts are the normative encoding sources. ISO 15924 supplies standardized script codes. Scholarly resources are retained as provenance-qualified sources for historical stages, transliteration and pronunciation.
