# Unicode ancient scripts and alphabet/sign sources

The ancient-language registry uses Unicode as the normative encoding layer and separates that layer from language identification.

## Registry inputs

- Unicode Character Database: character properties, names, scripts, bidi and normalization.
- Unicode code charts and NamesList: human-readable code-point/sign tables.
- Unicode Egyptian Hieroglyph Database (Unikemet): Egyptian sign metadata.
- ISO 15924: standardized script identifiers.
- Scholarly corpora: historical language, transliteration, lexicon and pronunciation assertions, retained with provenance.

The current inventory includes Greek, Greek Extended, Coptic, Hebrew, Samaritan, Imperial Aramaic, Syriac, Mandaic, Phoenician, Ugaritic, Palmyrene, Nabataean, Hatran, Old North Arabian, Old South Arabian, Cuneiform, Egyptian Hieroglyphs, Linear A/B, Cypriot and Old Italic. The registry is extensible to every Unicode script without changing application code.

## UTF-8

UTF-8 is derived deterministically from the Unicode scalar value. Stored bytes are validation data, not an independent authority. For example, Hebrew Alef `U+05D0` is `D7 90`; Greek alpha `U+03B1` is `CE B1`; Coptic Alfa `U+2C81` is `E2 B2 81`.

## Important distinction

A script inventory is not an alphabet inventory. Cuneiform and Egyptian hieroglyphs are sign systems with linguistic values and contextual readings. The application therefore uses `ALPHABET` for alphabetic systems and `GLYPH_SIGN`/`SIGN_VALUE` for logosyllabic systems.

## Release policy

Unicode 18.0.0 is currently shown by Unicode as a draft/beta-transition release in the September 2026 release cycle. The application records release status and source version in its snapshots rather than treating draft data as immutable. Once a final release is available, the pinned snapshot can be regenerated with the same deterministic tooling.
