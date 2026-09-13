# Ancient-language NLP and translation architecture

## Purpose

The scanner now treats translation as a pipeline rather than a single OCR button:

`image/text -> script recognition -> normalized source -> scholarly transliteration -> corpus lookup / NLP model -> target-language translation -> confidence + provenance`

The Python implementation remains CPU-friendly and does not require Tesseract or camel_tools.

## Ancient North Arabian / Thamudic

"Thamudic" is retained as a research label, but the application records the actual script variant when known. OCIANA notes that the historical label covers multiple Ancient North Arabian inscription groups and that Thamudic B/C/D remain incompletely classified. Every result therefore carries script, transliteration, translation status, confidence and provenance rather than collapsing all texts into one language model.

## Why the previous Translate action could appear empty

Recognition and transliteration are deterministic local operations in the current scanner. Translation needs a parallel text, lexicon/rule engine, or trained model. The old UI had no translation service boundary, so clicking translation could not produce a target-language result.

The new `/translate` API and Python `translate()` service always return:

- source text
- script variant
- scholarly transliteration
- target language
- translation or an explicit `not_available` status
- confidence
- corpus identifier
- provenance URL

Unknown fragments are never silently translated by guessing.

## Corpus-backed seed data

The initial offline seed contains published OCIANA examples for Safaitic, Dadanitic and Thamudic B. It is intentionally small and auditable. Larger corpora can be added through `CorpusEntry` objects or a provider implementing `lookup()`.

## Ancient-language NLP research integration

The architecture is compatible with several established research directions:

- Deep-learning prediction for Thamudic inscriptions, including sequence/context prediction.
- RNN-based cuneiform transliteration and segmentation.
- Neural machine translation for Akkadian from Unicode cuneiform and scholarly transliteration.
- Parallel-corpus NMT for Sumerian/English.
- Retrieval-augmented ancient-language translation using lexicons, dictionaries, genre detection and related-text context.

These systems belong behind adapters because training data, licenses, language coverage and scholarly certainty vary considerably.

## Supported target languages in the deterministic baseline

- English (`en`)
- Arabic (`ar`)

Additional targets should be implemented through explicit translation providers rather than character-by-character substitution. Transliteration is a representation of the source reading; it is not itself a translation.

## Provenance policy

Every corpus-backed translation must retain its corpus identifier and source URL. Fragmentary, uncertain or unmatched readings remain visibly uncertain. Historical reconstruction and machine-generated suggestions must not be presented as equivalent to an edited scholarly translation.

## Research references

- OCIANA — Online Corpus of the Inscriptions of Ancient North Arabia: https://ociana.osu.edu/
- Predicting Thamudic inscriptions pre and post-sequence using deep learning (npj Heritage Science, 2025): https://www.nature.com/articles/s40494-025-02020-2
- Reading Akkadian cuneiform using natural language processing (Akkademia): https://pmc.ncbi.nlm.nih.gov/articles/PMC7592802/
- Translating Akkadian to English with neural machine translation: https://pmc.ncbi.nlm.nih.gov/articles/PMC10153418/
- CDLI Sumerian/English Machine Translation: https://github.com/cdli-gh/Machine-Translation
- Ancient Language Processing workshop resources: https://www.ancientnlp.com/
