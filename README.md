# NLP / Ancient North Arabian Research Toolkit

This repository contains Python, C++, .NET, Visual C++ and desktop/web implementations for Thamudic and Ancient North Arabian language research.

## ThamudicScan web application

Public deployment:

https://thamudicscan-s3wz30.public.builtwithrocket.new/

The hosted application provides a browser-accessible interface for the Thamudic / North Arabian research tooling maintained in this repository.

## Ancient North Arabian Unicode support

The repository includes a canonical registry at `data/ancient_north_arabian/alphabet.json` covering the Unicode Old North Arabian block `U+10A80–U+10A9F`, including encoded characters/numbers, Unicode code points, scholarly transliteration, exact UTF-8 bytes, Dadanitic encoding basis and variant-script metadata.

Unicode encodes Old North Arabian using Dadanitic forms. Variant historical forms are represented as variant/font metadata rather than fabricated Unicode code points.

## Implementations

- `cpp/thamudic/` — C++20 library and Unicode registry.
- `vcpp/` — Visual Studio native Windows desktop scanner.
- `dotnet/` — CLI, WPF desktop and web implementations. Core/CLI target `net48;net6.0`; WPF targets `net48;net6.0-windows`; web is pinned to `net6.0`.
- `python/thamudic/` — Python Unicode/UTF-8/transliteration API.
- `python/tests/` — Unicode and UTF-8 regression tests.
- `data/ancient_north_arabian/` — language-neutral canonical registry.
- `ThamudicScan/` — documentation and link to the public web deployment.
- `apple/` — SwiftUI/Xcode iOS/iPadOS and macOS application boundary.

## Apple applications

`apple/project.yml` is an XcodeGen specification with iOS and macOS targets. `apple/Sources/` contains the SwiftUI shell. On macOS, install Xcode/XcodeGen, run `xcodegen generate --spec apple/project.yml`, then build/archive/export through the Apple script. Heavy Python/ML processing remains a separate native/service boundary rather than being assumed to run inside the IPA.

## Chimera 128D + authenticated P2P

Research records can participate in the common Chimera 128D application profile: geometry, temporal state, observer/perspective, light/material response, events, objects, properties and interaction relationships, plus an extensible perception/cognition layer.

Optional P2P synchronization is authenticated and provenance-preserving. It supports capability exchange, request/response, pub/sub, snapshot/delta and content-addressed research objects. It does not transfer credentials, private keys or arbitrary executable payloads and does not authorize unsolicited network scanning.

See `docs/CHIMERA_128D_P2P_INTEGRATION.md`.

## Windows desktop

The WPF application includes an Ancient North Arabian registry browser, text extraction/transliteration and UTF-8 inspection. The native VC++ application remains a separate implementation and solution.

## Compatibility terminology

Modern .NET 6 is `net6.0`; .NET Framework targets use TFMs such as `net48`. There is no Microsoft target named “.NET Framework 6.0”.

## Licensing

The repository is licensed under GNU GPL v3 or later; the existing `LICENSE` file contains the full GPLv3 text. Third-party dependencies remain under their respective licenses.

## Sources

Unicode Standard 17.0, Old North Arabian block U+10A80–U+10A9F and the Unicode NamesList are the normative character/code-point sources used by this implementation.
