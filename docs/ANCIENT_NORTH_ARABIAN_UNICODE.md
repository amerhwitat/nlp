# Ancient North Arabian Unicode / UTF-8 Registry

## Canonical repertoire

Unicode defines Old North Arabian in the range **U+10A80–U+10A9F**. The repertoire is based on the Dadanitic form. Unicode documents variant forms associated with Minaic, Safaitic, Hismaic, Taymanitic and Thamudic B; these variants should be rendered with appropriate fonts rather than assigned invented code points.

The block contains 29 letters and three numbers (one, ten and twenty). It is right-to-left in its encoded Dadanitic form.

## Repository representation

The canonical machine-readable source is:

`data/ancient_north_arabian/alphabet.json`

Each character contains:

- Unicode code point
- literal Unicode character
- Unicode name
- transliteration
- exact UTF-8 bytes in hexadecimal

Language-native projections live under:

- `cpp/thamudic/include/thamudic/old_north_arabian.hpp`
- `dotnet/src/Thamudic.Core/OldNorthArabian.cs`
- `python/thamudic/old_north_arabian.py`

## Example

`𐪀` is U+10A80 OLD NORTH ARABIAN LETTER HEH, transliterated `h`, and encoded in UTF-8 as `F0 90 AA 80`.

## Research sources

- Unicode Standard 17.0, Chapter 10, Old North Arabian: https://www.unicode.org/versions/Unicode17.0.0/core-spec/chapter-10/
- Unicode Old North Arabian NamesList: https://www.unicode.org/charts/nameslist/n_10A80.html
- Unicode Old North Arabian chart: https://www.unicode.org/charts/PDF/U10A80.pdf

## Scope note

The registry intentionally distinguishes **Unicode encoding** from **epigraphic glyph variation**. Dadanitic, Safaitic, Hismaic, Taymanitic, Minaic and Thamudic B may have visually different letter forms even where they correspond to the same Unicode character semantics.
