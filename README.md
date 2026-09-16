# NLP / Ancient North Arabian Research Toolkit

This repository contains Python, C++, .NET, Visual C++ and desktop/web implementations for Thamudic and Ancient North Arabian language research.

## Complete source-code citation index

| Area | Source |
|---|---|
| C++ Thamudic | [cpp/thamudic/](cpp/thamudic/) |
| Visual C++ | [vcpp/](vcpp/) |
| .NET | [dotnet/](dotnet/) |
| Python Thamudic | [python/thamudic/](python/thamudic/) |
| **Python all-in-one scanner** | **[python/ThamudicScanner_AllInOne.py](python/ThamudicScanner_AllInOne.py)** |
| **Python all-in-one NLP/media scanner** | **[python/NLPScanner_AllInOne.py](python/NLPScanner_AllInOne.py)** |
| **Image/PDF media pipeline** | [python/thamudic/media_pipeline.py](python/thamudic/media_pipeline.py) |
| Python translation/NLP | [python/thamudic/ancient_translation.py](python/thamudic/ancient_translation.py) |
| Universal ancient translation facade | [python/thamudic/universal_translation.py](python/thamudic/universal_translation.py) |
| Translation history/audit logger | [python/thamudic/translation_log.py](python/thamudic/translation_log.py) |
| PDF report/export engine | [python/thamudic/pdf_export.py](python/thamudic/pdf_export.py) |
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
| Apple | [apple/](apple/) |
| **Amiga Web Emulator** | **[Amiga/](Amiga/)** |

## Amiga Web Emulator

`Amiga/` is the browser-based Amiga emulation integration for the repository. It provides an Amiga-style web console, model/PAL-NTSC selection, ROM/floppy/hardfile pickers, memory controls, input hooks, diagnostics and an adapter for a locally vendored open-source emulator core.

The implementation was researched against Scripted Amiga Emulator (SAE), vAmigaWeb, UAE and FS-UAE. SAE is specifically designed for HTML5/JavaScript browser emulation and documents Amiga models, 68000-family CPUs, OCS/ECS/AGA, PAL/NTSC, Canvas/WebGL, WebAudio, keyboard/mouse/gamepad and disk-image support. vAmigaWeb exposes a C++ Amiga core to JavaScript/WebAssembly. The Library's existing Chimera II Web OS research also specifies an Amiga browser profile and browser-sandbox security model.

Start the web shell with:

```bash
python -m http.server 8080 --directory Amiga
```

The upstream fetch scripts are intentionally separate from the UI so that open-source emulator code can be reviewed and updated under its own license terms:

```bash
./Amiga/fetch_upstream.sh
# or
./Amiga/fetch_upstream.ps1
```

Kickstart ROMs and commercial Amiga software are not included. Users provide files for which they have the necessary rights.

## All-in-one Python scanner

`python/ThamudicScanner_AllInOne.py` consolidates the Python scanner runtime into one executable/copyable file. It includes Unicode and UTF-8 inspection, Old North Arabian extraction/transliteration, evidence-aware corpus translation, ancient-script metadata/capability reporting, translation-history logging and verification, PDF generation, optional voice capability detection, Tkinter GUI controls, and CLI operation.

Run the GUI with:

```bash
python python/ThamudicScanner_AllInOne.py --gui
```

## Media import: image/PDF → OCR/text → transliteration → translation

Both the modular Thamudic GUI and the standalone NLP scanner now support importing images, PDFs and text files. Text PDFs are extracted with `pypdf`. Images and scanned PDFs can use the optional EasyOCR + pypdfium2 pipeline. After extraction/OCR, the same scanner passes the resulting text through script detection, transliteration and the evidence-backed translation layer.

```bash
python python/NLPScanner_AllInOne.py --gui
python python/NLPScanner_AllInOne.py inscription.pdf --script Dadanitic --target en
```

For images:

```bash
python python/NLPScanner_AllInOne.py inscription.jpg --script Dadanitic --target en
```

OCR confidence/provider metadata is retained. OCR is not treated as proof of an ancient reading. When the extracted reading is not represented by an attested corpus entry or configured translation provider, the application explicitly reports translation unavailable instead of inventing a translation.

## Complete script report and export

The script-report layer combines one selected script/language with the actual source material. A report can preserve original characters, Unicode/script metadata, writing direction, historical variants, approximate dating, geographic scope, materials, related scripts, transliteration-system metadata, actual transliteration, actual corpus/model translation, target language, confidence, provider and provenance.

FastAPI endpoints:

- `GET /script-summary/{language}`
- `GET /script-summary/{language}/export?format=json|md|txt|pdf`
- `POST /script-report`
- `POST /script-report/export` with `format=json|md|txt|pdf`

PDF generation is provided by `python/thamudic/pdf_export.py` using ReportLab Platypus.

## Translation, transliteration and provenance

The Python scanner has separate translation and transliteration layers. The deterministic baseline supports its documented English/Arabic targets and corpus seed records. Unsupported fragments remain explicitly unavailable rather than receiving fabricated output.

The universal facade accepts `script`, `transliteration`, and `translation` source forms and can delegate to a real corpus/model provider. Every universal translation call can be persisted to an append-only JSON Lines audit log containing source, target, transliteration, translation, status, confidence, provider, provenance, script metadata, request metadata, timestamp and SHA-256 record hash.

### Translation history

Default log:

```text
translation_logs/translations.jsonl
```

Override with `THAMUDIC_TRANSLATION_LOG`.

API:

- `GET /translation-log`
- `GET /translation-log/export?format=json|jsonl|txt|pdf`

The integrity checker detects modified records by recomputing each record's deterministic SHA-256 hash. This is an integrity aid, not a substitute for signed or immutable archival storage.

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

## Methodological note

“Thamudic” is retained as a user-facing research category, but the implementation records script variants explicitly. Unicode identifies encoded characters/scripts, not proof of a particular language. Translation requires an attested corpus, lexicon or trained model; unsupported reverse translations remain retrieval/model tasks rather than character substitution.
