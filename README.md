# NLP / Ancient North Arabian Research Toolkit

This repository contains Python, C++, .NET, Visual C++ and desktop/web implementations for Thamudic and Ancient North Arabian language research.

## ThamudicScan web application

Public deployment:

https://thamudicscan-s3wz30.public.builtwithrocket.new/

The hosted application provides a browser-accessible interface for the Thamudic / North Arabian research tooling maintained in this repository.

## Ancient North Arabian Unicode support

The repository now includes a canonical registry at `data/ancient_north_arabian/alphabet.json` covering the complete Unicode Old North Arabian block `U+10A80–U+10A9F`:

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
- `dotnet/` — CLI, WPF desktop and web implementations. Core/CLI target `net48;net6.0`; WPF targets `net48;net6.0-windows`; web is pinned to `net6.0` because ASP.NET Core is a modern .NET runtime.
- `python/thamudic/` — Python Unicode/UTF-8/transliteration API.
- `python/tests/` — Unicode and UTF-8 regression tests.
- `data/ancient_north_arabian/` — language-neutral canonical registry.
- `ThamudicScan/` — documentation and link to the public web deployment.
- `chimera/` — shared 128D/P2P interoperability contract.

## Chimera multidimensional + P2P layer

Research records can carry the common Chimera 128D state model, including geometry, time, observer/perspective, light/material response, events, objects, properties, interactions and extensible vector/cognitive state. `chimera/p2p_protocol.json` defines authenticated peer exchange, capability discovery, pub/sub, request/response, replay controls and content-addressed synchronization. Peer discovery is configured rather than arbitrary Internet scanning.

## Windows desktop

The WPF application includes an Ancient North Arabian registry browser, text extraction/transliteration and UTF-8 inspection. The native VC++ application remains a separate implementation and solution.

## Compatibility terminology

Modern .NET 6 is `net6.0`; .NET Framework targets use TFMs such as `net48`. There is no Microsoft target named “.NET Framework 6.0”.

## Sources

Unicode Standard 17.0, Old North Arabian block U+10A80–U+10A9F and the Unicode NamesList are the normative character/code-point sources used by this implementation.

## License

Original project code is released under the GNU General Public License v3 or later. Third-party components retain their applicable licenses and notices.
