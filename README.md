# NLP / Ancient North Arabian Research Toolkit

This repository contains Python, C++, .NET, Visual C++ and desktop/web implementations for Thamudic and Ancient North Arabian language research.

## Complete source-code citation index

| Area | Source |
|---|---|
| C++ Thamudic | [cpp/thamudic/](cpp/thamudic/) |
| Visual C++ | [vcpp/](vcpp/) |
| .NET | [dotnet/](dotnet/) |
| Python Thamudic | [python/thamudic/](python/thamudic/) |
| Python translation/NLP | [python/thamudic/ancient_translation.py](python/thamudic/ancient_translation.py) |
| Python source-language scanner | [python/thamudic/source_language_scanner.py](python/thamudic/source_language_scanner.py) |
| Python tests | [python/tests/](python/tests/) |
| Ancient/classical Unicode registry | [data/source_languages/ancient_classical_unicode.json](data/source_languages/ancient_classical_unicode.json) |
| Ancient-language NLP documentation | [docs/ANCIENT_LANGUAGE_NLP_TRANSLATION.md](docs/ANCIENT_LANGUAGE_NLP_TRANSLATION.md) |
| Ancient North Arabian registry | [data/ancient_north_arabian/alphabet.json](data/ancient_north_arabian/alphabet.json) |
| ThamudicScan web frontend | [ThamudicScan/web_ui/](ThamudicScan/web_ui/) |
| ThamudicScan FastAPI backend | [ThamudicScan/server/](ThamudicScan/server/) |
| Web API documentation | [ThamudicScan/docs/WEB_API.md](ThamudicScan/docs/WEB_API.md) |
| Web architecture | [ThamudicScan/docs/ARCHITECTURE.md](ThamudicScan/docs/ARCHITECTURE.md) |
| ThamudicScan product docs | [ThamudicScan/](ThamudicScan/) |
| Apple | [apple/](apple/) |
| Complete tracked repository | [source tree](.) |

## Translation and transliteration

The Python scanner has a real translation service boundary. The UI action **Translate + transliterate** calls the FastAPI `/translate` endpoint and displays both outputs independently. The deterministic baseline supports English and Arabic targets and uses corpus-backed OCIANA seed records. When a fragment has no supported parallel reading, the result explicitly says `not_available` instead of inventing a translation.

Translation results retain script variant, corpus identifier, confidence and provenance. Transliteration is kept separate from translation because a scholarly transliteration is a representation of the reading, not a target-language translation.

## Ancient Egyptian, Chinese, Japanese, Greek and Latin source scanner

The repository now includes a language-oriented Unicode/UTF-8 scanner registry at `data/source_languages/ancient_classical_unicode.json` and Python implementation at `python/thamudic/source_language_scanner.py` for:

- **Ancient Egyptian** — Egyptian Hieroglyphs `U+13000–U+1342F` and Egyptian Hieroglyphs Extended-A `U+13460–U+143FF`.
- **Chinese** — Han/CJK Unified Ideographs and their encoded extensions.
- **Japanese** — Hiragana, Katakana, Kana extensions, and Han/Kanji coverage.
- **Greek / Ancient Greek** — Greek `U+0370–U+03FF` and Greek Extended `U+1F00–U+1FFF`.
- **Latin / Classical Latin** — Basic Latin plus relevant Latin Extended and scholarly ranges.

For every matched character the scanner exposes Unicode code point, Unicode name, NFC form, UTF-8 hexadecimal bytes and byte array. It also reports language-profile counts and script overlap. This is deliberately a scanner rather than a false language classifier: Unicode encodes scripts/characters, not languages, and Han is shared by Chinese and Japanese.

The FastAPI endpoint `POST /scan_language` and web UI expose the same implementation so desktop/Python and browser workflows share one registry and one UTF-8 policy.

## ThamudicScan web application

The browser stack is split into:

- `ThamudicScan/web_ui/` — React + Vite interface with scanning, validation, source-language Unicode/UTF-8 scanning, translation, transliteration, target-language selection and export.
- `ThamudicScan/server/` — FastAPI API with `/scan`, `/validate`, `/translate`, `/scan_language`, file upload, persistence, SSE progress and exporters.
- `ThamudicScan/server/tests/` — pytest contracts.

The web layer reuses `python/thamudic` rather than copying Old North Arabian mapping tables.

## Ancient North Arabian Unicode support

The repository includes a canonical registry at `data/ancient_north_arabian/alphabet.json` covering the Unicode Old North Arabian block `U+10A80–U+10A9F`, including encoded characters/numbers, Unicode code points, scholarly transliteration, exact UTF-8 bytes, Dadanitic encoding basis and variant-script metadata.

## Ancient-language NLP research

The repository documents an adapter architecture for Thamudic sequence prediction, cuneiform transliteration/segmentation, Akkadian machine translation, Sumerian-English NMT, lexicon retrieval and human-in-the-loop scholarly correction. These model integrations are kept separate from the deterministic scanner so that uncertainty and provenance remain visible.

## Implementations

- `cpp/thamudic/` — C++20 library and Unicode registry.
- `vcpp/` — Visual Studio native Windows desktop scanner.
- `dotnet/` — CLI, WPF desktop and web implementations.
- `python/thamudic/` — Python Unicode/UTF-8/transliteration, translation API and universal source-language scanner.
- `python/tests/` — Unicode, transliteration, translation and source-language scanner regression tests.
- `data/ancient_north_arabian/` — language-neutral Ancient North Arabian registry.
- `data/source_languages/` — language-oriented Unicode/UTF-8 scanner registry.
- `ThamudicScan/` — React/FastAPI web application and documentation.
- `apple/` — existing SwiftUI/Xcode application boundary.

## Methodological note

“Thamudic” is retained as a user-facing research category, but the implementation records script variants explicitly. OCIANA notes that the historical Thamudic label covers multiple Ancient North Arabian groups and that some categories remain incompletely classified. The application therefore preserves the distinction between script identification, transliteration, translation and scholarly uncertainty.

Unicode similarly encodes scripts rather than languages. The new Ancient Egyptian/Chinese/Japanese/Greek/Latin scanner therefore reports Unicode evidence and overlap rather than asserting language identity from a single character.

## Licensing

The repository is licensed under GNU GPL v3 or later; the existing `LICENSE` file contains the full GPLv3 text. Third-party dependencies remain under their respective licenses.
