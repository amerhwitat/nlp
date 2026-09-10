# Thamudic language layout

```text
thamudic/
  python/legacy/       Python reference applications and datasets
  cpp/thamudic/        C++20 portable core
  java/thamudic/       Java 21 core
  node/thamudic/       Node.js ESM core
  vcpp/                 Visual C++ / MSVC x64 application
  dotnet/
    src/Thamudic.Core/  C# core for .NET 8/9/10
    src/Thamudic.Cli/   .NET CLI
  tests/                cross-language fixtures
  docs/                 specifications and interoperability
```

The existing Python scripts are intentionally preserved. The new implementations are parallel language targets sharing the same Unicode range, transliteration semantics and scanner data contracts.
