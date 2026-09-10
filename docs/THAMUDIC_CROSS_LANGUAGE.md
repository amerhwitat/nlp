# Thamudic cross-language implementation

The Python Thamudic/North Arabian research stack is now mirrored by language-specific implementations while the Python originals remain backward-compatible.

## Layout

```text
nlp/
  python/                         # future consolidated Python package; legacy scripts remain at root
  cpp/thamudic/                   # C++20 portable core + CTest
  java/thamudic/                  # Java 21 Maven module + JUnit
  node/thamudic/                  # Node.js ESM package + node:test
  vcpp/ThamudicScanner.sln       # Visual C++ / MSVC x64 solution
  dotnet/
    src/Thamudic.Core/            # C# multi-target library: net8.0;net9.0;net10.0
    src/Thamudic.Cli/             # cross-version CLI
  docs/                            # interoperability and research documentation
```

## Shared behavior

- Ancient North Arabian / Thamudic Unicode range U+10A80..U+10A9F.
- Unicode code-point extraction and transliteration-map support.
- Connected-component primitives and line/word grouping for scanner pipelines.
- UTF-8/UTF-32-safe processing.
- English, Arabic and Hebrew text can remain in surrounding metadata without being forced through the Thamudic classifier.
- The architecture does not require Tesseract or camel_tools; OCR/model adapters are optional external layers.
- Safaitic, Hismaic, Dadanitic and Early Arabic remain dataset/classifier extensions rather than being incorrectly treated as identical glyph inventories.

## Python-to-native mapping

| Python responsibility | C++ | Java | Node.js | C#/.NET |
|---|---|---|---|---|
| Unicode detection | `isThamudic` | `isThamudic` | `isThamudic` | `IsThamudic` |
| sequence extraction | `extractThamudic` | `extract` | `extract` | `Extract` |
| transliteration | `transliterate` | `transliterate` | `transliterate` | `Transliterate` |
| image components | `connectedComponents` | `connectedComponents` | `connectedComponents` | `Box` foundation |

The original Python files such as `thamudic.py`, `thamudic-scanner.py`, `thamudic_desktop.py`, `thamudic_web_app.py`, and related tests remain available. This migration is additive rather than destructive.

## .NET policy

The C# library targets .NET 8, 9 and 10. Microsoft currently lists .NET 10 as LTS and .NET 8 as LTS, while .NET 9 is STS; the project therefore multi-targets all three and allows deployment policy to select the supported runtime appropriate to the host.

## Build

- C++: CMake 3.20+, C++20, CTest.
- Java: JDK 21+, Maven.
- Node: Node.js 20+ with ESM and `node:test`.
- C#: `dotnet build` for net8.0/net9.0/net10.0.
- Visual C++: Visual Studio/MSVC v143, x64.

## Conformance goal

All language implementations should consume the same Unicode fixtures, transliteration maps, bounding-box fixtures and JSON interchange envelopes. GUI applications can layer on top of these cores without duplicating scanner logic.
