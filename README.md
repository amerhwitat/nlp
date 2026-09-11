# NLP / Ancient North Arabian Research Toolkit

This repository contains Python, C++, .NET, Visual C++ and desktop/web implementations for Thamudic and Ancient North Arabian language research.

## Chimera II Mobile FlashKit

`FlashTool/` adds a safety-first Android device inspection and authorized firmware flashing subsystem. It provides a canonical C++ core with C, Python, Java and architecture-specific assembly interfaces, plus GUI/build documentation.

Core workflow: **discover -> inspect -> validate -> dry-run -> explicit confirmation -> flash -> verify**.

The FlashTool does not bypass OEM bootloader authorization, FRP, credentials, Android Verified Boot, secure boot or vendor security controls. Vendor-specific protocols are supported only when their interfaces are documented and authorized.

See `FlashTool/README.md` and `FlashTool/markdown.md` for architecture, supported tooling and build information.

## Ancient North Arabian Unicode support

The repository now includes a canonical registry at `data/ancient_north_arabian/alphabet.json` covering the complete Unicode Old North Arabian block `U+10A80–U+10A9F`.

- 29 letters
- 3 encoded numbers
- Unicode character
- Unicode code point
- scholarly transliteration
- exact UTF-8 byte sequence
- Dadanitic encoding basis
- variant-script metadata for Safaitic, Hismaic, Hismaic, Taymanitic, Minaic and Thamudic B

## Implementations

- `cpp/thamudic/` — C++20 library and Unicode registry.
- `vcpp/` — Visual Studio native Windows desktop scanner.
- `dotnet/` — CLI, WPF desktop and web implementations.
- `python/thamudic/` — Python Unicode/UTF-8/transliteration API.
- `python/tests/` — Unicode and UTF-8 regression tests.
- `data/ancient_north_arabian/` — language-neutral canonical registry.
- `FlashTool/` — Chimera II Android device inspection/flashing toolkit.

## Compatibility terminology

Modern .NET 6 is `net6.0`; .NET Framework targets use TFMs such as `net48`.

## Sources

Unicode Standard 17.0 and the Unicode NamesList are the normative character/code-point sources used by the language implementation. Android tooling is expected to be installed from its official distribution rather than copied into this repository.
