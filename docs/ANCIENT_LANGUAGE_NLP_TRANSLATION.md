# Ancient-language NLP, Unicode scanning, and translation architecture

## Purpose

The scanner now treats recognition, transliteration and translation as separate evidence layers:

`source text/image -> script/language profile -> Unicode normalization -> UTF-8 inspection -> scholarly transliteration -> corpus lookup / NLP model -> target-language translation -> confidence + provenance`

The Python implementation remains CPU-friendly and does not require Tesseract or camel_tools.

## Requested source-language scanner coverage

The canonical registry is `data/source_languages/ancient_classical_unicode.json`. It currently provides scanner profiles for:

| Source-language research category | Principal Unicode coverage | Notes |
|---|---|---|
| Ancient Egyptian | Egyptian Hieroglyphs `U+13000-U+1342F`; Egyptian Hieroglyphs Extended-A `U+13460-U+143FF` | Hieroglyphic source text; Hieratic/Demotic are not incorrectly assumed to be the same Unicode script inventory |
| Chinese | Han/CJK Unified Ideographs and extensions | Han is shared across languages; language identification is contextual |
| Japanese | Hiragana, Katakana, Kana extensions, plus Han/Kanji | Japanese commonly mixes multiple scripts |
| Greek / Ancient Greek | Greek `U+0370-U+03FF`; Greek Extended `U+1F00-U+1FFF` | Greek Extended is important for polytonic scholarly text |
| Latin / Classical Latin | Basic Latin plus Latin Extended and scholarly Latin ranges | Latin script is shared by many languages; the profile is a research category for Latin-language source text |

Unicode encodes scripts and characters rather than languages. The implementation therefore reports script/range matches and exposes ambiguity instead of pretending that every character uniquely identifies a language. This is especially important for Han characters shared by Chinese and Japanese. See the Unicode Consortium's supported-scripts documentation: https://www.unicode.org/standard/supported.html

For Ancient Egyptian, the registry follows the current Unicode Egyptian Hieroglyph database model. Unicode documents Egyptian Hieroglyphs at `U+13000-U+1342F` and Egyptian Hieroglyphs Extended-A at `U+13460-U+143FF`; the Unikemet data is UTF-8/NFC. Reference: https://www.unicode.org/reports/tr57/tr57-7.html

## UTF-8 / Unicode inspection

`python/thamudic/source_language_scanner.py` provides:

- Unicode code point (`U+...`)
- decimal scalar value
- official Unicode character name where available
- UTF-8 hexadecimal bytes
- UTF-8 byte array
- NFC-normalized character
- matched source-language profiles
- per-language counts
- overlap/ambiguity flag

The scanner does not convert characters into legacy 8-bit encodings. UTF-8 is the canonical interchange representation. Unicode blocks are organizational ranges, not separate encodings.

## API

The FastAPI service exposes:

- `POST /translate` — Ancient North Arabian corpus-backed transliteration/translation.
- `POST /scan_language` — Unicode/UTF-8 source-language scanning for Ancient Egyptian, Chinese, Japanese, Greek and Latin.
- `POST /validate` — existing Old North Arabian validation.
- `POST /scan` and `POST /scan_file` — existing scanner workflows.

Example request:

```json
{
  "text": "ἄνθρωπος",
  "language": "greek"
}
```

The response includes detected profiles, counts, Unicode character metadata and UTF-8 bytes for each matched character.

## Web UI

`ThamudicScan/web_ui/` now includes a **Source-language Unicode scanner** with:

- Auto detect
- Ancient Egyptian
- Chinese
- Japanese
- Greek / Ancient Greek
- Latin / Classical Latin

The existing Ancient North Arabian translation/transliteration controls remain separate, because a Unicode scanner is not itself a translation model.

## Ancient North Arabian / Thamudic

"Thamudic" is retained as a research label, but the application records the actual script variant when known. OCIANA notes that the historical label covers multiple Ancient North Arabian inscription groups and that Thamudic B/C/D remain incompletely classified. Every result therefore carries script, transliteration, translation status, confidence and provenance rather than collapsing all texts into one language model.

## Why the previous Translate action could appear empty

Recognition and transliteration are deterministic local operations in the current scanner. Translation needs a parallel text, lexicon/rule engine, or trained model. The old UI had no translation service boundary, so clicking translation could not produce a target-language result.

The `/translate` API and Python `translate()` service always return source text, script variant, scholarly transliteration, target language, translation or explicit `not_available`, confidence, corpus identifier and provenance URL. Unknown fragments are never silently translated by guessing.

## Corpus-backed seed data

The initial offline seed contains published OCIANA examples for Safaitic, Dadanitic and Thamudic B. It is intentionally small and auditable. Larger corpora can be added through `CorpusEntry` objects or a provider implementing `lookup()`.

## Ancient-language NLP research integration

The architecture is compatible with established research directions including deep-learning sequence prediction for Thamudic, RNN-based cuneiform transliteration/segmentation, neural machine translation for Akkadian, parallel-corpus NMT for Sumerian/English, and retrieval/lexicon-assisted ancient-language translation.

These systems belong behind adapters because training data, licenses, language coverage and scholarly certainty vary considerably.

## Supported target languages in the deterministic baseline

- English (`en`)
- Arabic (`ar`)

Additional targets should be implemented through explicit translation providers rather than character-by-character substitution. Transliteration is a representation of the source reading; it is not itself a translation.

## Provenance policy

Every corpus-backed translation must retain its corpus identifier and source URL. Fragmentary, uncertain or unmatched readings remain visibly uncertain. Historical reconstruction and machine-generated suggestions must not be presented as equivalent to an edited scholarly translation.

## Research references

- Unicode Supported Scripts: https://www.unicode.org/standard/supported.html
- Unicode Egyptian Hieroglyph Database (Unikemet): https://www.unicode.org/reports/tr57/tr57-7.html
- OCIANA — Online Corpus of the Inscriptions of Ancient North Arabia: https://ociana.osu.edu/
- Predicting Thamudic inscriptions pre and post-sequence using deep learning (npj Heritage Science, 2025): https://www.nature.com/articles/s40494-025-02020-2
- Reading Akkadian cuneiform using natural language processing (Akkademia): https://pmc.ncbi.nlm.nih.gov/articles/PMC7592802/
- Translating Akkadian to English with neural machine translation: https://pmc.ncbi.nlm.nih.gov/articles/PMC10153418/
- CDLI Sumerian/English Machine Translation: https://github.com/cdli-gh/Machine-Translation
- Ancient Language Processing workshop resources: https://www.ancientnlp.com/
