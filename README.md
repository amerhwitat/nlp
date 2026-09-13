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
| Translation history/audit logger | [python/thamudic/translation_log.py](python/thamudic/translation_log.py) |
| Script summary/report exporter | [python/thamudic/script_summary.py](python/thamudic/script_summary.py) |
| Voice/TTS/STT capability layer | [python/thamudic/voice.py](python/thamudic/voice.py) |
| Python source-language scanner | [python/thamudic/source_language_scanner.py](python/thamudic/source_language_scanner.py) |
| Python ancient alphabet registry | [python/thamudic/ancient_alphabet_registry.py](python/thamudic/ancient_alphabet_registry.py) |
| Python tests | [python/tests/](python/tests/) |
| Ancient/classical Unicode registry | [data/source_languages/ancient_classical_unicode.json](data/source_languages/ancient_classical_unicode.json) |
| Ancient alphabet/variation registry | [data/source_languages/ancient_language_alphabets.json](data/source_languages/ancient_language_alphabets.json) |
| Historical script metadata/dating | [data/source_languages/ancient_script_metadata.json](data/source_languages/ancient_script_metadata.json) |
| Ancient-language NLP documentation | [docs/ANCIENT_LANGUAGE_NLP_TRANSLATION.md](docs/ANCIENT_LANGUAGE_NLP_TRANSLATION.md) |
| Translation history documentation | [docs/TRANSLATION_HISTORY.md](docs/TRANSLATION_HISTORY.md) |
| Research/interoperability sources | [docs/RESEARCH_SOURCES.md](docs/RESEARCH_SOURCES.md) |
| Ancient North Arabian registry | [data/ancient_north_arabian/alphabet.json](data/ancient_north_arabian/alphabet.json) |
| ThamudicScan web frontend | [ThamudicScan/web_ui/](ThamudicScan/web_ui/) |
| ThamudicScan FastAPI backend | [ThamudicScan/server/](ThamudicScan/server/) |
| Web API documentation | [ThamudicScan/docs/WEB_API.md](ThamudicScan/docs/WEB_API.md) |
| Web architecture | [ThamudicScan/docs/ARCHITECTURE.md](ThamudicScan/docs/ARCHITECTURE.md) |
| ThamudicScan product docs | [ThamudicScan/](ThamudicScan/) |
| Apple | [apple/](apple/) |

## Complete script report and export

The script-report layer combines one selected script/language with the actual source material. A report can preserve original characters, Unicode/script metadata, writing direction, historical variants, approximate dating, geographic scope, materials, related scripts, transliteration-system metadata, actual transliteration, actual corpus/model translation, target language, confidence, provider and provenance.

FastAPI endpoints:

- `GET /script-summary/{language}`
- `GET /script-summary/{language}/export?format=json|md|txt`
- `POST /script-report`
- `POST /script-report/export`

Historical dates are intentionally broad research metadata and do not override inscription-specific palaeographic or archaeological dating.

## Translation, transliteration and provenance

The Python scanner has separate translation and transliteration layers. The deterministic baseline supports its documented English/Arabic targets and corpus seed records. Unsupported fragments remain explicitly unavailable rather than receiving fabricated output.

The universal facade accepts `script`, `transliteration`, and `translation` source forms and can delegate to a real corpus/model provider. Every universal translation call can now be persisted to an append-only JSON Lines audit log containing source, target, transliteration, translation, status, confidence, provider, provenance, script metadata, request metadata, timestamp and SHA-256 record hash.

### Translation history

Default log:

```text
translation_logs/translations.jsonl
```

Override with `THAMUDIC_TRANSLATION_LOG`.

API:

- `GET /translation-log`
- `GET /translation-log/export?format=json|jsonl|txt`

The integrity checker detects modified records by recomputing each record's deterministic SHA-256 hash. This is an integrity aid, not a substitute for signed or immutable archival storage.

See [docs/TRANSLATION_HISTORY.md](docs/TRANSLATION_HISTORY.md).

## Voice / speech capabilities

- Browser Speech Synthesis playback.
- Original, transliteration and translation playback.
- Pause, resume and stop.
- Voice capability discovery.
- Optional Python `pyttsx3` local TTS.
- Speech-recognition capability reporting.

Native ancient pronunciation is treated as a separate scholarly provider/model problem. The application does not silently use a modern voice and label it as an authenticated ancient pronunciation.

## Ancient alphabet, script and historical-variation registry

The registry covers Ancient Egyptian, Akkadian, Sumerian, Ugaritic, Phoenician/Punic, Ancient/Paleo-Hebrew, Aramaic families, Ancient North Arabian and Old South Arabian, Ancient Greek, Latin, historical Chinese, historical Japanese, Old Persian, Sanskrit, Coptic, Hittite, Luwian, Etruscan, Gothic, Old Turkic, Linear B/Mycenaean Greek and Cypro-Minoan.

## Ancient-language source interoperability

Research adapters are designed around public corpus conventions such as OCIANA and ORACC. OCIANA provides Ancient North Arabian readings, translations, commentary, bibliography, provenance and images; ORACC provides structured ATF/JSON conventions for cuneiform language and text editions. External corpus/model adapters remain provenance-aware and license-aware.

See [docs/RESEARCH_SOURCES.md](docs/RESEARCH_SOURCES.md).

## ThamudicScan web application

The browser stack is split into `ThamudicScan/web_ui/` and `ThamudicScan/server/`. The FastAPI service provides scanning, validation, translation, universal translation, source-language scanning, language registry, script reports, voice capabilities, translation history, exports, persistence and SSE progress.

## Ancient North Arabian Unicode support

The repository includes a canonical registry at `data/ancient_north_arabian/alphabet.json` covering the Unicode Old North Arabian block `U+10A80–U+10A9F`, with encoded characters, code points, scholarly transliteration, UTF-8 bytes and variant-script metadata.

## Methodological note

“Thamudic” is retained as a user-facing research category, but the implementation records script variants explicitly. Unicode identifies encoded characters/scripts, not proof of a particular language. Translation requires an attested corpus, lexicon or trained model; unsupported reverse translations remain retrieval/model tasks rather than character substitution.

## Licensing

The repository is licensed under GNU GPL v3 or later; the existing `LICENSE` file contains the full GPLv3 text. Third-party dependencies remain under their respective licenses.
