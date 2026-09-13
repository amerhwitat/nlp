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
| Python tests | [python/tests/](python/tests/) |
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

The Python scanner now has a real translation service boundary. The UI action **Translate + transliterate** calls the FastAPI `/translate` endpoint and displays both outputs independently. The deterministic baseline supports English and Arabic targets and uses corpus-backed OCIANA seed records. When a fragment has no supported parallel reading, the result explicitly says `not_available` instead of inventing a translation.

Translation results retain script variant, corpus identifier, confidence and provenance. Transliteration is kept separate from translation because a scholarly transliteration is a representation of the reading, not a target-language translation.

## ThamudicScan web application

The browser stack is split into:

- `ThamudicScan/web_ui/` — React + Vite interface with scanning, validation, translation, transliteration, target-language selection and export.
- `ThamudicScan/server/` — FastAPI API with `/scan`, `/validate`, `/translate`, file upload, persistence, SSE progress and exporters.
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
- `python/thamudic/` — Python Unicode/UTF-8/transliteration and translation API.
- `python/tests/` — Unicode, transliteration and translation regression tests.
- `data/ancient_north_arabian/` — language-neutral canonical registry.
- `ThamudicScan/` — React/FastAPI web application and documentation.
- `apple/` — existing SwiftUI/Xcode application boundary.

## Methodological note

“Thamudic” is retained as a user-facing research category, but the implementation records script variants explicitly. OCIANA notes that the historical Thamudic label covers multiple Ancient North Arabian groups and that some categories remain incompletely classified. The application therefore preserves the distinction between script identification, transliteration, translation and scholarly uncertainty.

## Licensing

The repository is licensed under GNU GPL v3 or later; the existing `LICENSE` file contains the full GPLv3 text. Third-party dependencies remain under their respective licenses.
