# Ancient-language NLP, Unicode scanning, alphabets, and translation architecture

## Purpose

The toolkit separates recognition, alphabet/script identification, transliteration and translation into evidence layers:

`source text/image -> script/language profile -> Unicode normalization -> UTF-8 inspection -> alphabet/variant identification -> scholarly transliteration -> corpus/lexicon/NLP model -> target-language translation -> confidence + provenance`

The Python implementation remains CPU-friendly and does not require Tesseract or camel_tools.

## Ancient alphabet and historical-variation registry

The canonical alphabet metadata is `data/source_languages/ancient_language_alphabets.json`, exposed through `python/thamudic/ancient_alphabet_registry.py`.

Current registry families include Ancient Egyptian; Akkadian; Sumerian; Ugaritic; Phoenician/Punic; Ancient/Paleo-Hebrew; Aramaic families; Ancient North Arabian; Old South Arabian; Ancient Greek; Latin; historical Chinese; historical Japanese; Old Persian; Sanskrit; Coptic; Hittite; Luwian; Etruscan; Gothic; Old Turkic; Linear B/Mycenaean Greek; and Cypro-Minoan, with documented historical/script variations in each profile.

The registry records language identifiers, scripts, directionality, Unicode blocks and translation modes. It is an extensible foundation for additional ancient scripts and variants.

## Translation directions

The architecture distinguishes four directions:

1. **Source script → scholarly transliteration** — deterministic or OCR/model-assisted character/sign reading.
2. **Transliteration → target-language translation** — corpus, dictionary, grammar or trained NLP model.
3. **Source script/text → target-language translation** — combined reading + translation pipeline.
4. **Target-language → source-script retrieval/generation** — retrieval from attested corpus first; generative reconstruction must be explicitly labeled and confidence-scored.

`python/thamudic/universal_translation.py` implements the provider contract for these directions. It accepts `script`, `transliteration`, or `translation` as the source form and returns a structured result containing source language, target language, transliteration, status, confidence, provider and provenance.

A registered capability does **not** mean that a local model is installed. Without an applicable corpus/model provider the result is `provider_required`; unsupported directions are `direction_not_registered`. This prevents an alphabet table from being mistaken for a translation engine.

## Unicode and UTF-8

Unicode explicitly encodes scripts/characters rather than languages. The scanner therefore reports script/range evidence and ambiguity rather than asserting language identity from an isolated character. This is particularly important for Han, which is shared by Chinese and Japanese.

The application records Unicode code points, official names, normalized text and UTF-8 hexadecimal/byte-array representations. UTF-8 is the canonical interchange representation.

## Source-language scanner

`data/source_languages/ancient_classical_unicode.json` and `python/thamudic/source_language_scanner.py` provide Unicode/UTF-8 scanning for Ancient Egyptian, Chinese, Japanese, Greek/Ancient Greek and Latin/Classical Latin. The broader alphabet registry adds historical variants and additional ancient languages without conflating script identification with language translation.

## API

The FastAPI service exposes:

- `POST /translate` — existing evidence-backed Ancient North Arabian translation service.
- `POST /translate_ancient` — universal provider-facing ancient-language translation contract.
- `GET /translation-matrix` — directional capability matrix for all registered languages.
- `POST /scan_language` — Unicode/UTF-8 source-language scanning.
- `GET /alphabet-languages` — all registered language IDs.
- `GET /alphabet-languages/{language}` — alphabet/script variants, Unicode blocks and translation capabilities for one language.
- `POST /validate` — Old North Arabian validation.
- `POST /scan` and `POST /scan_file` — scanner workflows.

Example universal request:

```json
{
  "text": "ἄνθρωπος",
  "source_language": "ancient-greek",
  "source_form": "script",
  "target_language": "en"
}
```

A provider-backed installation can return an attested or model-supported translation. The base installation returns an explicit provider status instead of hallucinating one.

## Scholarly safety and provenance

A historical alphabet is not automatically a language dictionary, grammar or translation model. The implementation therefore distinguishes:

- Unicode character evidence
- script identification
- historical variant
- transliteration
- lexical/grammatical interpretation
- attested translation
- machine-generated suggestion

Only attested or model-supported translations should be emitted as translations. Unknown material remains `not_available` or `provider_required` rather than being filled with guessed text. Reverse translation should prefer retrieval from attested corpora before any generative reconstruction.

## Research integration

The architecture can host adapters for cuneiform transliteration/segmentation, Akkadian NMT, Sumerian-English NMT, Ancient North Arabian sequence prediction, Ancient Egyptian text models, historical Chinese/Japanese reading models, Greek/Latin corpora, lexicon retrieval and human scholarly correction. Training corpora, licenses and uncertainty metadata remain attached to each adapter.

## References

- Unicode Supported Scripts: https://www.unicode.org/standard/supported.html
- Unicode Character Code Charts: https://www.unicode.org/charts/
- Unicode Egyptian Hieroglyph Database (Unikemet): https://www.unicode.org/reports/tr57/tr57-7.html
- OCIANA — Online Corpus of the Inscriptions of Ancient North Arabia: https://ociana.osu.edu/
- Predicting Thamudic inscriptions pre and post-sequence using deep learning: https://www.nature.com/articles/s40494-025-02020-2
- Reading Akkadian cuneiform using natural language processing: https://pmc.ncbi.nlm.nih.gov/articles/PMC7592802/
- Translating Akkadian to English with neural machine translation: https://pmc.ncbi.nlm.nih.gov/articles/PMC10153418/
- CDLI Sumerian/English Machine Translation: https://github.com/cdli-gh/Machine-Translation
- Ancient Language Processing workshop resources: https://www.ancientnlp.com/
