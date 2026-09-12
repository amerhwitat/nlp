# Thamudic Cross-Language Epigraphy Platform

A research-oriented integration layer for the Thamudic/Ancient North Arabian scanner and related ancient-language research workflows. It combines provenance-aware databases, FastAPI APIs, OCR, language registries, translation proofing, speech proofing, chatbot orchestration, PDF research exchange, KPI dashboards and cross-language clients.

## Architecture

```text
Historical PDFs / inscriptions / source records
                    |
        quality + geometry analysis
                    |
 RTL/LTR/TTB/BTT | spiral | reverse | skew | weathering
                    |
       OCR adapters + confidence/provenance
                    |
 Unicode/BCP47/CLDR language registry
                    |
 transliteration -> literal/meaning/interlinear/scholarly
                    |
 proof -> speech/phonemes -> research chatbot
                    |
 FastAPI -> React/TypeScript -> Python/C++/C#/Java/Go/Rust/JS/TS
                    |
 SQLite | PostgreSQL | MySQL | JSON | CSV | Access/ODBC
```

## Features

- UTF-8 Old North Arabian/Thamudic support (`U+10A80-U+10A9F`).
- Ancient-language registry architecture for Mesopotamia, Egypt, Arabia, Greek, Latin and adjacent Levantine languages.
- Chinese and Japanese as both source and target languages, including classical/vertical-writing metadata.
- BCP-47/CLDR target-language resolution instead of a closed target-language list.
- OCR geometry hypotheses for LTR, RTL, top-to-bottom, bottom-to-top, spiral, reverse, skewed and weathered material.
- Intelligent OCR with Kraken/Tesseract adapters plus quality-only mode.
- Unicode NFC normalization, source hashes, script candidates, bounding boxes, confidence and warnings.
- Literal, meaning, interlinear and scholarly translation contracts.
- Proofing layer preserving alternatives, uncertainty and human-review requirements.
- Model-neutral RNN/Transformer/LLM speech proof contract; reconstructed ancient pronunciation is explicitly labelled.
- Provider-neutral research chatbot API with evidence/citation context.
- Historical-object PDF import/export and KPI dashboards.
- SQLite canonical database plus PostgreSQL and MySQL portability schemas.
- JSON and CSV/TSV flat-file interchange; optional Microsoft Access through `pyodbc` and a locally installed ODBC driver.
- Cross-language API clients and build automation.
- Windows CMD, PowerShell, Bash and Docker dependency/build/deployment automation.

## OCR geometry API

`POST /api/ocr/geometry` accepts image dimensions and optional quality/orientation hints and returns a conservative routing hypothesis and adapter options. The heuristic does not claim to identify an ancient writing system; it prepares the image pipeline.

PaddleOCR documents orientation classification, text-line orientation and document unwarping, and its current multilingual documentation includes Chinese, Traditional Chinese and Japanese. See [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR).

## OCR scanner

`POST /api/ocr/scan` accepts a bounded image upload and returns recognition text, confidence, script candidates, bounding boxes, image-quality metrics, warnings, preprocessing and SHA-256 provenance.

Supported engines: `auto`, `kraken`, `tesseract`, `quality-only`.

## Translation and proofing

`POST /api/translation/proof` packages an engine-produced translation as either:

- **literal** — close lexical/syntactic rendering;
- **meaning** — context-aware interpreted sense;
- **interlinear** — aligned source/gloss/translation;
- **scholarly** — provenance, alternatives and confidence.

The API does not invent a translation when no model supplies one.

## Speech proofing

`POST /api/speech/proof` records language, voice profile, optional phonemes, model type, confidence and whether pronunciation is reconstructed. Empty phoneme data is preserved rather than guessed.

## Research chatbot

`POST /api/chat` is a provider-neutral orchestration boundary. A deployment can attach a self-hosted/local LLM, RAG system or another approved provider. Evidence and citations can be passed in `context`. The default implementation fails safely when no model adapter is configured.

Research references include [Rasa](https://github.com/RasaHQ/rasa) and [LangChain](https://github.com/langchain-ai/langchain). Botpress is not treated as a current self-hosted open-source dependency.

## Data exchange and SQL

Canonical database: SQLite. Portability schemas are provided for PostgreSQL and MySQL. `tools/data_exchange.py` supports JSON and CSV output and optional Access export through `pyodbc`.

## Dependency automation

```bash
./scripts/check-dependencies.sh
# optional OCR extras:
INSTALL_OCR_EXTRAS=1 ./scripts/check-dependencies.sh
```

Windows:

```powershell
$env:INSTALL_OCR_EXTRAS='1'; .\scripts\check-dependencies.ps1
```

```bat
set INSTALL_OCR_EXTRAS=1
scripts\check-dependencies.bat
```

## Research citations

- [Unicode supported scripts](https://www.unicode.org/standard/supported.html)
- [Unicode 18.0](https://www.unicode.org/versions/Unicode18.0.0/)
- [CLDR](https://cldr.unicode.org/)
- [BCP47 extensions](https://cldr.unicode.org/index/bcp47-extension)
- [Unicode Transliteration Guidelines](https://cldr.unicode.org/index/cldr-spec/transliteration-guidelines)
- [RFC 6497](https://www.rfc-editor.org/rfc/rfc6497)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [Kraken](https://github.com/mittagessen/kraken)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [Tesseract](https://github.com/tesseract-ocr/tesseract)
- [Cuneiform sign detection](https://github.com/CompVis/cuneiform-sign-detection-code)
- [CuReD](https://github.com/DigitalPasts/CuReD)
- [pypdf](https://github.com/py-pdf/pypdf)
- [ReportLab](https://www.reportlab.com/)
- [JSON Schema](https://json-schema.org/)

See `docs/SOURCES.md` and `docs/OCR_GEOMETRY_TRANSLATION_CHAT.md` for detailed provenance and licensing notes.
