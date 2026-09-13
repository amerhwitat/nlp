# NLP / Ancient North Arabian Research Toolkit

This repository contains Python, C++, .NET, Visual C++ and desktop/web implementations for Thamudic and Ancient North Arabian language research.

## Complete source-code citation index

| Area | Source |
|---|---|
| C++ Thamudic | [cpp/thamudic/](cpp/thamudic/) |
| Visual C++ | [vcpp/](vcpp/) |
| .NET | [dotnet/](dotnet/) |
| Python Thamudic | [python/thamudic/](python/thamudic/) |
| Python translation/NLP | [python/thamudic/ancient_translation.py](python/thamudic/ancient_translation.py) |
| Universal ancient translation facade | [python/thamudic/universal_translation.py](python/thamudic/universal_translation.py) |
| Script summary/report exporter | [python/thamudic/script_summary.py](python/thamudic/script_summary.py) |
| Voice/TTS/STT capability layer | [python/thamudic/voice.py](python/thamudic/voice.py) |
| Python source-language scanner | [python/thamudic/source_language_scanner.py](python/thamudic/source_language_scanner.py) |
| Python ancient alphabet registry | [python/thamudic/ancient_alphabet_registry.py](python/thamudic/ancient_alphabet_registry.py) |
| Python tests | [python/tests/](python/tests/) |
| Ancient/classical Unicode registry | [data/source_languages/ancient_classical_unicode.json](data/source_languages/ancient_classical_unicode.json) |
| Ancient alphabet/variation registry | [data/source_languages/ancient_language_alphabets.json](data/source_languages/ancient_language_alphabets.json) |
| Historical script metadata/dating | [data/source_languages/ancient_script_metadata.json](data/source_languages/ancient_script_metadata.json) |
| Ancient-language NLP documentation | [docs/ANCIENT_LANGUAGE_NLP_TRANSLATION.md](docs/ANCIENT_LANGUAGE_NLP_TRANSLATION.md) |
| Ancient North Arabian registry | [data/ancient_north_arabian/alphabet.json](data/ancient_north_arabian/alphabet.json) |
| ThamudicScan web frontend | [ThamudicScan/web_ui/](ThamudicScan/web_ui/) |
| ThamudicScan FastAPI backend | [ThamudicScan/server/](ThamudicScan/server/) |
| Web API documentation | [ThamudicScan/docs/WEB_API.md](ThamudicScan/docs/WEB_API.md) |
| Web architecture | [ThamudicScan/docs/ARCHITECTURE.md](ThamudicScan/docs/ARCHITECTURE.md) |
| ThamudicScan product docs | [ThamudicScan/](ThamudicScan/) |
| Apple | [apple/](apple/) |

## Complete script report and export

The new script-report layer combines **one selected script/language** with the actual source material. A report can preserve:

1. Original source characters exactly as entered.
2. Unicode/script metadata and script family.
3. Writing direction and a human-readable direction description.
4. Historical variants and orthographic forms.
5. Approximate dating and dating confidence/status.
6. Geographic scope and writing materials.
7. Related/ancestor/sister script relationships.
8. Unicode blocks and code-point information.
9. Scholarly transliteration-system metadata.
10. Corpus/model translation, when available.
11. Translation target language, provider, confidence and provenance.
12. Explicit unavailable/uncertain status instead of fabricated translation.

FastAPI endpoints:

- `GET /script-summary/{language}` — metadata-only script summary.
- `GET /script-summary/{language}/export?format=json|md|txt` — metadata export.
- `POST /script-report` — combines original text, transliteration, translation and script metadata.
- `POST /script-report/export` — exports the complete report as JSON, Markdown or TXT.

Historical dates are intentionally broad research metadata. They do not override inscription-specific palaeographic or archaeological dating.

## Translation and transliteration

The Python scanner has a real translation service boundary. The UI action **Translate + transliterate** calls the FastAPI `/translate` endpoint and displays both outputs independently. The deterministic baseline supports English and Arabic targets and uses corpus-backed OCIANA seed records. When a fragment has no supported parallel reading, the result explicitly says `not_available` instead of inventing a translation.

Translation results retain script variant, corpus identifier, confidence and provenance. Transliteration is kept separate from translation because a scholarly transliteration is a representation of the reading, not a target-language translation.

## Universal ancient-language translation architecture

`python/thamudic/universal_translation.py` provides a provider-oriented facade for the entire alphabet/variation registry. A provider can implement:

`original script -> scholarly transliteration -> target language`

and, where attested resources permit:

`target language -> source-script retrieval`

The facade supports `script`, `transliteration`, and `translation` source forms, returns confidence/provider/provenance metadata, and reports `provider_required` when a registered direction has no local corpus/model. It never fabricates an ancient-language translation merely from an alphabet table.

## Voice / speech capabilities

Voice support is now exposed through both the web and Python desktop applications.

### Web

- Browser Speech Synthesis playback.
- Original-script, transliteration and translation playback controls.
- Pause, resume and stop.
- Voice capability discovery.
- Optional backend fallback through the API.
- Speech-recognition capability reporting.

### Python desktop

- Optional `pyttsx3` local TTS.
- Original/transliteration/translation voice controls.
- Pause/resume/stop controls with backend-specific behavior.
- No silent substitution of a modern voice for an ancient language.

Ancient-script **native pronunciation is a separate scholarly model/provider**. When one is not installed, the application explicitly returns `pronunciation_provider_required`; it does not pretend that an English/Arabic TTS engine knows an ancient pronunciation.

## Ancient alphabet, script and historical-variation registry

The `data/source_languages/ancient_language_alphabets.json` registry provides common metadata for Ancient Egyptian, Akkadian, Sumerian, Ugaritic, Phoenician/Punic, Ancient/Paleo-Hebrew, Aramaic families, Ancient North Arabian and Old South Arabian, Ancient Greek, Latin, historical Chinese, historical Japanese, Old Persian, Sanskrit, Coptic, Hittite, Luwian, Etruscan, Gothic, Old Turkic, Linear B/Mycenaean Greek and Cypro-Minoan.

`data/source_languages/ancient_script_metadata.json` adds dating, dating uncertainty, regions, script type, writing materials, related scripts, transliteration systems and scholarly notes for every registry entry.

## Ancient Egyptian, Chinese, Japanese, Greek and Latin source scanner

The repository includes a language-oriented Unicode/UTF-8 scanner registry at `data/source_languages/ancient_classical_unicode.json` and Python implementation at `python/thamudic/source_language_scanner.py` for Ancient Egyptian, Chinese, Japanese, Greek/Ancient Greek and Latin/Classical Latin. For every matched character the scanner exposes Unicode code point, Unicode name, NFC form, UTF-8 hexadecimal bytes and byte array. It also reports language-profile counts and script overlap.

This is deliberately a scanner rather than a false language classifier: Unicode encodes scripts/characters, not languages, and Han is shared by Chinese and Japanese.

## ThamudicScan web application

The browser stack is split into:

- `ThamudicScan/web_ui/` — React + Vite interface with scanning, validation, source-language Unicode/UTF-8 scanning, language/variant selection, translation, transliteration, complete script reports, JSON/Markdown/TXT export and voice controls.
- `ThamudicScan/server/` — FastAPI API with `/scan`, `/validate`, `/translate`, `/translate_ancient`, `/scan_language`, `/alphabet-languages`, `/translation-matrix`, `/script-summary`, `/script-report`, `/voice/capabilities`, `/voice/speak`, file upload, persistence, SSE progress and exporters.
- `ThamudicScan/server/tests/` — pytest contracts.

## Ancient North Arabian Unicode support

The repository includes a canonical registry at `data/ancient_north_arabian/alphabet.json` covering the Unicode Old North Arabian block `U+10A80–U+10A9F`, including encoded characters/numbers, Unicode code points, scholarly transliteration, exact UTF-8 bytes, Dadanitic encoding basis and variant-script metadata.

## Ancient-language NLP research

The repository documents an adapter architecture for Thamudic sequence prediction, cuneiform transliteration/segmentation, Akkadian machine translation, Sumerian-English NMT, lexicon retrieval and human-in-the-loop scholarly correction. These model integrations are kept separate from the deterministic scanner so that uncertainty and provenance remain visible.

## Methodological note

“Thamudic” is retained as a user-facing research category, but the implementation records script variants explicitly. OCIANA notes that the historical Thamudic label covers multiple Ancient North Arabian groups and that some categories remain incompletely classified. The application therefore preserves the distinction between script identification, transliteration, translation and scholarly uncertainty.

Unicode similarly encodes scripts rather than languages. The alphabet registry consequently records script/language relationships and historical variants without treating a Unicode block as proof of language identity. Translation requires a corpus, lexicon or trained model; unsupported reverse translations remain retrieval/model tasks rather than fabricated character substitutions.

## Licensing

The repository is licensed under GNU GPL v3 or later; the existing `LICENSE` file contains the full GPLv3 text. Third-party dependencies remain under their respective licenses.
