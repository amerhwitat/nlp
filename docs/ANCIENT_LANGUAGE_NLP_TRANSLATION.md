# Ancient-language NLP, Unicode scanning, alphabets, and translation architecture

## Purpose

The toolkit separates recognition, alphabet/script identification, transliteration and translation into evidence layers:

`source text/image -> script/language profile -> Unicode normalization -> UTF-8 inspection -> alphabet/variant identification -> scholarly transliteration -> corpus/lexicon/NLP model -> target-language translation -> confidence + provenance`

The Python implementation remains CPU-friendly and does not require Tesseract or camel_tools.

## Ancient alphabet and historical-variation registry

The canonical alphabet metadata is `data/source_languages/ancient_language_alphabets.json`, exposed through `python/thamudic/ancient_alphabet_registry.py`.

Current registry families include:

| Language | Script/alphabet families | Historical or orthographic variations |
|---|---|---|
| Ancient Egyptian | Egyptian Hieroglyphs | Old, Middle, Late Egyptian; Hieratic; Demotic; hieroglyphic writing |
| Akkadian | Sumero-Akkadian Cuneiform | Old Akkadian; Old/Middle/Neo-Babylonian; Old/Middle/Neo-Assyrian |
| Sumerian | Sumero-Akkadian Cuneiform | Early Dynastic; Ur III; Old Babylonian literary; Neo-Sumerian |
| Ugaritic | Ugaritic alphabetic cuneiform | Ugaritic alphabetic cuneiform |
| Phoenician | Phoenician | Phoenician; Punic; Neo-Punic |
| Ancient Hebrew | Hebrew / Paleo-Hebrew | Paleo-Hebrew; Biblical Hebrew; Second Temple Hebrew |
| Aramaic | Imperial Aramaic and related scripts | Imperial/Biblical Aramaic; Palmyrene; Nabataean; Hatran; Syriac |
| Ancient North Arabian | Old North Arabian | Safaitic; Dadanitic; Hismaic; Taymanitic; Thamudic B/C/D |
| Old South Arabian | Old South Arabian | Sabaean; Minaean; Qatabanian; Hadramitic |
| Ancient Greek | Greek | Mycenaean/Linear B; Archaic; Classical; Koine/Hellenistic |
| Latin | Latin / Old Italic | Old Latin; Classical; Late; Medieval; epigraphic Latin |
| Historical Chinese | Han | Oracle Bone; Bronze; Seal; Clerical; Traditional Han; Literary/Classical Chinese |
| Historical Japanese | Han + Hiragana + Katakana | Man'yogana; Hentaigana; Kanbun; Classical Japanese; Kyujitai; historical kana |
| Old Persian | Old Persian cuneiform | Achaemenid Old Persian |
| Sanskrit | Brahmi-derived scripts | Vedic; Classical; Brahmi; Devanagari; Grantha; Sharada; Siddham |
| Coptic | Coptic | Sahidic; Bohairic; Fayyumic; Akhmimic |
| Hittite | Cuneiform / Anatolian Hieroglyphs | Old; Middle; New Hittite |
| Luwian | Anatolian Hieroglyphs / Cuneiform | Hieroglyphic Luwian; Cuneiform Luwian |
| Etruscan | Old Italic | Etruscan alphabetic inscriptions |
| Gothic | Gothic | Gothic alphabet |
| Old Turkic | Old Turkic | Orkhon; Yenisei |
| Mycenaean Greek | Linear B | Syllabary; ideograms |
| Cypro-Minoan | Cypro-Minoan | Cypro-Minoan |

The registry records language identifiers, scripts, directionality, Unicode blocks and translation modes. It is an extensible foundation for additional ancient scripts and variants.

## Translation directions

The architecture supports four distinct directions where evidence/models exist:

1. **Source script → scholarly transliteration** — deterministic or OCR/model-assisted character/sign reading.
2. **Transliteration → target-language translation** — corpus, dictionary, grammar or trained NLP model.
3. **Source script/text → target-language translation** — combined reading + translation pipeline.
4. **Target-language → source-script retrieval/generation** — retrieval from attested corpus first; generative reconstruction must be explicitly labeled and confidence-scored.

The registry exposes these as capabilities rather than claiming every language has a production model. An alphabet table alone cannot translate a language.

## Unicode and UTF-8

Unicode explicitly encodes scripts/characters rather than languages. The scanner therefore reports script/range evidence and ambiguity rather than asserting language identity from an isolated character. This is particularly important for Han, which is shared by Chinese and Japanese. See the Unicode Consortium's supported-scripts documentation: https://www.unicode.org/standard/supported.html

Unicode 17.0/18.0 code charts include the major ancient-script blocks used by this project, including Egyptian Hieroglyphs, Cuneiform, Early Dynastic Cuneiform, Ugaritic, Old Persian, Old North Arabian, Old South Arabian, Phoenician, Hebrew, Imperial Aramaic, Greek, Latin, Old Italic, Gothic, Old Turkic, Linear B, Cypro-Minoan and CJK/Han extensions. See the Unicode script charts: https://www.unicode.org/charts/

For Ancient Egyptian, Unicode documents Egyptian Hieroglyphs at `U+13000-U+1342F` and Egyptian Hieroglyphs Extended-A at `U+13460-U+143FF`. Reference: https://www.unicode.org/reports/tr57/tr57-7.html

UTF-8 remains the canonical interchange representation. The application records UTF-8 hexadecimal bytes and byte arrays instead of inventing legacy encodings for Unicode blocks.

## Requested source-language scanner coverage

The existing `data/source_languages/ancient_classical_unicode.json` scanner provides profiles for Ancient Egyptian, Chinese, Japanese, Greek/Ancient Greek and Latin/Classical Latin. The broader alphabet registry now extends the metadata layer to the additional ancient languages listed above.

## API

The FastAPI service exposes:

- `POST /translate` — existing evidence-backed Ancient North Arabian translation service.
- `POST /scan_language` — Unicode/UTF-8 source-language scanning.
- `GET /alphabet-languages` — all registered language IDs.
- `GET /alphabet-languages/{language}` — alphabet/script variants, Unicode blocks and translation capabilities for a language.
- `POST /validate` — Old North Arabian validation.
- `POST /scan` and `POST /scan_file` — scanner workflows.

The new registry endpoints make the complete alphabet/variation table available to desktop and web implementations through one source of truth.

## Scholarly safety and provenance

A historical alphabet is not automatically a language dictionary, grammar or translation model. The implementation therefore distinguishes:

- Unicode character evidence
- script identification
- historical variant
- transliteration
- lexical/grammatical interpretation
- attested translation
- machine-generated suggestion

Only attested or model-supported translations should be emitted as translations. Unknown material remains `not_available` rather than being filled with guessed text. Reverse translation should prefer retrieval from attested corpora before any generative reconstruction.

## Research integration

The architecture can host adapters for cuneiform transliteration/segmentation, Akkadian NMT, Sumerian-English NMT, Ancient North Arabian sequence prediction, lexicon retrieval and human scholarly correction. Training corpora, licenses and uncertainty metadata remain attached to each adapter.

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
