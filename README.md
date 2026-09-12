# NLP / Ancient North Arabian Research Toolkit

This repository contains Python, C++, .NET, Visual C++ and desktop/web implementations for Thamudic and Ancient North Arabian language research.

## Complete source-code citation index

| Area | Source |
|---|---|
| C++ Thamudic | [cpp/thamudic/](cpp/thamudic/) |
| Visual C++ | [vcpp/](vcpp/) |
| .NET | [dotnet/](dotnet/) |
| Python Thamudic | [python/thamudic/](python/thamudic/) |
| Python tests | [python/tests/](python/tests/) |
| Ancient North Arabian registry | [data/ancient_north_arabian/alphabet.json](data/ancient_north_arabian/alphabet.json) |
| ThamudicScan web frontend | [ThamudicScan/web_ui/](ThamudicScan/web_ui/) |
| ThamudicScan FastAPI backend | [ThamudicScan/server/](ThamudicScan/server/) |
| Web API documentation | [ThamudicScan/docs/WEB_API.md](ThamudicScan/docs/WEB_API.md) |
| Web architecture | [ThamudicScan/docs/ARCHITECTURE.md](ThamudicScan/docs/ARCHITECTURE.md) |
| ThamudicScan product docs | [ThamudicScan/](ThamudicScan/) |
| Apple | [apple/](apple/) |
| Complete tracked repository | [source tree](.) |

These links are the README-level citations for all maintained implementation areas; component READMEs remain the detailed file-level source record.

## Centralized Apple Objective-C + Flutter implementation

The Apple companion is maintained in [`general/Apple-Implementations/nlp`](https://github.com/amerhwitat/general/tree/master/Apple-Implementations/nlp). It provides Objective-C/Xcode native integration and Flutter iOS/macOS UI. Heavy Python/ML workloads remain behind an explicit native/service boundary.

## ThamudicScan web application

Public deployment is documented in `ThamudicScan/`; deployment credentials are never stored in source control.

The new browser stack is split into:

- `ThamudicScan/web_ui/` — React + Vite responsive interface with keyword search, inscription/text input, upload/drop-zone, live progress, statistics, Unicode-aware results, session reopen and CSV/JSON export.
- `ThamudicScan/server/` — FastAPI API, scanner adapter, SQLite session/result/event persistence, ordered SSE progress stream, upload validation and exporters.
- `ThamudicScan/server/tests/` — pytest contracts for Unicode validation, canonical transliteration integration, persistence, exports and HTTP endpoints.

The web layer reuses `python/thamudic` rather than copying its Old North Arabian mapping tables.

## Ancient North Arabian Unicode support

The repository includes a canonical registry at `data/ancient_north_arabian/alphabet.json` covering the Unicode Old North Arabian block `U+10A80–U+10A9F`, including encoded characters/numbers, Unicode code points, scholarly transliteration, exact UTF-8 bytes, Dadanitic encoding basis and variant-script metadata.

## Implementations

- `cpp/thamudic/` — C++20 library and Unicode registry.
- `vcpp/` — Visual Studio native Windows desktop scanner.
- `dotnet/` — CLI, WPF desktop and web implementations.
- `python/thamudic/` — Python Unicode/UTF-8/transliteration API.
- `python/tests/` — Unicode and UTF-8 regression tests.
- `data/ancient_north_arabian/` — language-neutral canonical registry.
- `ThamudicScan/` — React/FastAPI web application and documentation.
- `apple/` — existing SwiftUI/Xcode application boundary.

## Apple applications

Use Xcode/XcodeGen on macOS to generate the native shell. Flutter provides the UI while Objective-C provides Apple framework access and high-performance native services.

## Chimera 128D + authenticated P2P

Research records can participate in the common Chimera 128D application profile: geometry, temporal state, observer/perspective, light/material response, events, objects, properties and interaction relationships, plus an extensible perception/cognition/vector layer.

Optional P2P synchronization is authenticated and provenance-preserving. It supports capability exchange, request/response, pub/sub, snapshot/delta and content-addressed research objects. It does not transfer credentials, private keys or arbitrary executable payloads and does not authorize unsolicited network scanning.

## Licensing

The repository is licensed under GNU GPL v3 or later; the existing `LICENSE` file contains the full GPLv3 text. Third-party dependencies remain under their respective licenses.
