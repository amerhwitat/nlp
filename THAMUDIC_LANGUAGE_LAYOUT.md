# Thamudic / Ancient North Arabian language layout

```text
nlp/
  data/ancient_north_arabian/  canonical Unicode + UTF-8 registry
  python/thamudic/             Python Unicode/transliteration API
  python/tests/                Python interoperability tests
  cpp/thamudic/                C++20 portable core + ONA registry
  vcpp/                        Visual C++ / MSVC x64 desktop application
  dotnet/
    src/Thamudic.Core/         C# core: net48 + net6.0
    src/Thamudic.Cli/          CLI: net48 + net6.0
    src/Thamudic.Desktop/      WPF: net48 + net6.0-windows
    src/Thamudic.Web/          ASP.NET Core: net6.0
```

## Unicode contract

The canonical repertoire is `U+10A80–U+10A9F`. It is encoded in Unicode using Dadanitic forms. Safaitic, Hismaic, Taymanitic, Minaic and Thamudic B are represented as documented variant forms rather than assigned invented code points.

All implementations expose the same code-point, character, transliteration and UTF-8-byte semantics. The Python, C++ and .NET implementations are deliberately maintained in their respective language directories.
