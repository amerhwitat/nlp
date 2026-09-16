# NLP / Ancient North Arabian Research Toolkit

This repository contains Python, C++, .NET, Visual C++ and desktop/web implementations for Thamudic and Ancient North Arabian language research.

## AI disciplines layer

The repository now includes `python/ai_disciplines/` and `cpp/ai/discipline_registry.hpp`, providing a provider-neutral contract for Machine Learning, Deep Learning, Reinforcement Learning, Symbolic AI, Computer Vision and NLP. The ancient-language/OCR pipeline can therefore combine deterministic script metadata with optional statistical, neural, visual, symbolic and language-model stages without treating model output as scholarly proof.

Reference provider families include scikit-learn, PyTorch, TensorFlow/JAX, Gymnasium, Stable-Baselines3, SymPy, OpenCV/scikit-image, spaCy and Hugging Face Transformers. Optional providers are imported lazily; unavailable dependencies are reported rather than silently replaced.

The intended flow is **image/PDF → CV/OCR → script detection → NLP/tokenization → transliteration → symbolic/corpus checks → translation/model stage → provenance/audit report**.

## Complete source-code citation index

| Area | Source |
|---|---|
| C++ Thamudic | [cpp/thamudic/](cpp/thamudic/) |
| C++ AI discipline registry | [cpp/ai/discipline_registry.hpp](cpp/ai/discipline_registry.hpp) |
| Visual C++ | [vcpp/](vcpp/) |
| .NET | [dotnet/](dotnet/) |
| Python Thamudic | [python/thamudic/](python/thamudic/) |
| Python AI discipline registry | [python/ai_disciplines/discipline_registry.py](python/ai_disciplines/discipline_registry.py) |
| Python AI pipeline dispatcher | [python/ai_disciplines/pipeline.py](python/ai_disciplines/pipeline.py) |
| Python all-in-one scanner | [python/ThamudicScanner_AllInOne.py](python/ThamudicScanner_AllInOne.py) |
| Python all-in-one NLP/media scanner | [python/NLPScanner_AllInOne.py](python/NLPScanner_AllInOne.py) |
| Image/PDF media pipeline | [python/thamudic/media_pipeline.py](python/thamudic/media_pipeline.py) |
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
| Amiga Web Emulator | [Amiga/](Amiga/) |

## Existing Amiga Web Emulator

`Amiga/` provides the browser-based Amiga emulation integration with model/PAL-NTSC selection, ROM/floppy/hardfile pickers, memory controls, input hooks, diagnostics and an adapter for an open-source emulator core. Upstream fetch scripts remain separate so licenses and provenance can be reviewed before importing source.

## All-in-one Python scanner

`python/ThamudicScanner_AllInOne.py` consolidates the Python scanner runtime into one executable/copyable file. It includes Unicode and UTF-8 inspection, Old North Arabian extraction/transliteration, evidence-aware corpus translation, ancient-script metadata/capability reporting, translation-history logging and verification, PDF generation, optional voice capability detection, Tkinter GUI controls, and CLI operation.

Run the GUI with:

```bash
python python/ThamudicScanner_AllInOne.py --gui
```

## Media import: image/PDF → OCR/text → transliteration → translation

Both the modular Thamudic GUI and standalone NLP scanner support importing images, PDFs and text files. Text PDFs are extracted with `pypdf`. Images and scanned PDFs can use the optional EasyOCR + pypdfium2 pipeline. After extraction/OCR, the scanner passes the resulting text through script detection, transliteration and evidence-backed translation.

```bash
python python/NLPScanner_AllInOne.py --gui
python python/NLPScanner_AllInOne.py inscription.pdf --script Dadanitic --target en
```

OCR confidence/provider metadata is retained. OCR is not treated as proof of an ancient reading. When the extracted reading is not represented by an attested corpus entry or configured translation provider, the application reports translation unavailable instead of inventing a translation.

## Script reports and exports

The script-report layer can preserve original characters, Unicode/script metadata, writing direction, historical variants, approximate dating, geographic scope, materials, related scripts, transliteration metadata, actual transliteration, actual corpus/model translation, target language, confidence, provider and provenance.

FastAPI endpoints include `/script-summary/{language}`, `/script-report`, `/script-report/export`, `/translation-log` and `/translation-log/export`. PDF generation uses ReportLab Platypus.

## Translation, transliteration and provenance

The Python scanner has separate translation and transliteration layers. Unsupported fragments remain explicitly unavailable rather than receiving fabricated output. Universal translation calls can be persisted to an append-only JSON Lines audit log with source, target, transliteration, translation, status, confidence, provider, provenance, script metadata, request metadata, timestamp and SHA-256 record hash.

## Voice / speech capabilities

Browser Speech Synthesis, optional Python `pyttsx3`, speech-recognition capability reporting and original/transliteration/translation playback are supported. Native ancient pronunciation is treated as a separate scholarly provider/model problem and is not silently represented by a modern voice.

## Ancient alphabet and historical-variation registry

The registry covers Ancient Egyptian, Akkadian, Sumerian, Ugaritic, Phoenician/Punic, Ancient/Paleo-Hebrew, Aramaic families, Ancient North Arabian and Old South Arabian, Ancient Greek, Latin, historical Chinese/Japanese, Old Persian, Sanskrit, Coptic, Hittite, Luwian, Etruscan, Gothic, Old Turkic, Linear B/Mycenaean Greek and Cypro-Minoan.

## Methodological note

“Thamudic” is retained as a user-facing research category, but implementation records script variants explicitly. Unicode identifies encoded characters/scripts, not proof of a particular language. Translation requires an attested corpus, lexicon or trained model; unsupported reverse translations remain retrieval/model tasks rather than character substitution.
