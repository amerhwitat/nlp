# NLP / Ancient North Arabian Research Toolkit

This repository contains Python, C++, .NET, Visual C++ and desktop/web implementations for Thamudic and Ancient North Arabian language research.

## Centralized Apple Objective-C + Flutter implementation

The Apple companion is maintained in [`general/Apple-Implementations/nlp`](https://github.com/amerhwitat/general/tree/master/Apple-Implementations/nlp). It provides Objective-C/Xcode native integration and Flutter iOS/macOS UI. Heavy Python/ML workloads remain behind an explicit native/service boundary.

## ThamudicScan web application

Public deployment:

https://thamudicscan-s3wz30.public.builtwithrocket.new/

## Ancient North Arabian Unicode support

The repository includes a canonical registry at `data/ancient_north_arabian/alphabet.json` covering the Unicode Old North Arabian block `U+10A80–U+10A9F`, including encoded characters/numbers, Unicode code points, scholarly transliteration, exact UTF-8 bytes, Dadanitic encoding basis and variant-script metadata.

## Implementations

- `cpp/thamudic/` — C++20 library and Unicode registry.
- `vcpp/` — Visual Studio native Windows desktop scanner.
- `dotnet/` — CLI, WPF desktop and web implementations.
- `python/thamudic/` — Python Unicode/UTF-8/transliteration API.
- `python/tests/` — Unicode and UTF-8 regression tests.
- `data/ancient_north_arabian/` — language-neutral canonical registry.
- `ThamudicScan/` — documentation and link to the public web deployment.
- `apple/` — existing SwiftUI/Xcode application boundary.

## Apple applications

Use Xcode/XcodeGen on macOS to generate the native shell. Flutter provides the UI while Objective-C provides Apple framework access and high-performance native services. This structure follows Flutter's supported Objective-C platform-channel integration model for iOS and macOS. citeturn0search0

## Chimera 128D + authenticated P2P

Research records can participate in the common Chimera 128D application profile: geometry, temporal state, observer/perspective, light/material response, events, objects, properties and interaction relationships, plus an extensible perception/cognition layer.

Optional P2P synchronization is authenticated and provenance-preserving. It supports capability exchange, request/response, pub/sub, snapshot/delta and content-addressed research objects. It does not transfer credentials, private keys or arbitrary executable payloads and does not authorize unsolicited network scanning.

## Licensing

The repository is licensed under GNU GPL v3 or later; the existing `LICENSE` file contains the full GPLv3 text. Third-party dependencies remain under their respective licenses.
