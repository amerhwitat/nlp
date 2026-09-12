# OCR geometry, translation, speech and research chatbot

## OCR geometry

The scanner now treats document geometry as an explicit hypothesis rather than assuming horizontal Latin-like text. Supported routing labels are:

- LTR and RTL;
- top-to-bottom and bottom-to-top vertical text;
- spiral/path-based layout;
- reverse-reading hypothesis;
- skew/deskew and perspective unwarping;
- weathered/low-contrast preprocessing.

The implementation exposes these through `/api/ocr/geometry`. Engine adapters may replace the heuristic router with learned orientation/unwarping models without changing the API contract.

PaddleOCR documents dedicated document-orientation classification, text-line orientation and document-unwarping modules, and its current multilingual recognition documentation includes Chinese, Traditional Chinese and Japanese. Its current PP-OCRv5 documentation lists 106 supported languages. citeturn0search0turn0search1turn0search3

OpenCV perspective correction and deskew patterns are used as clean-room architectural references; the repository does not copy external implementation code. citeturn0search6

## Chinese and Japanese

Chinese and Japanese are registered as both source and target languages. The registry preserves script directionality because historical Chinese and Japanese materials may use vertical columns and right-to-left column ordering, while modern digital text is commonly horizontal.

Language targets resolve through BCP-47/CLDR rather than a closed list. The default target set includes English (`en`), Arabic (`ar`), Simplified/Modern Chinese (`zh`) and Japanese (`ja`).

## Translation modes

1. **Literal** — close lexical/syntactic rendering; uncertainty and unresolved readings remain visible.
2. **Meaning** — context-aware rendering intended to communicate the interpreted sense.
3. **Interlinear** — source, transliteration, gloss and translation aligned by token.
4. **Scholarly** — provenance, alternatives, notes and confidence retained.

The API never turns an OCR hypothesis into a historical fact automatically. `/api/translation/proof` records confidence, alternatives and review requirements.

## Neural/RNN/LLM proof and speech

The platform exposes proofing as a model-neutral boundary. A future RNN/Transformer/LLM adapter can supply candidate reading, morphology, contextual proofing and pronunciation/phoneme evidence. Speech records must retain language/voice profile, model provenance, confidence and audio hash. Reconstructed ancient pronunciation must be explicitly labelled as reconstructed rather than presented as a recorded historical fact.

## Chatbot

`server/chatbot.py` defines a provider-neutral research chatbot interface. It can be connected to a self-hosted Rasa/LangChain-style orchestration layer, an approved local LLM, or another API through an adapter. Evidence and citations travel with the response.

Rasa is a useful open/self-hosted conversational architecture reference; LangChain is a composable framework for LLM chains, retrieval, memory and tool use. Botpress is listed only as a historical ecosystem reference because its former open-source self-hosted v12 is sunset. citeturn0search5

## OCR engine references

- PaddleOCR: orientation, unwarping, multilingual recognition and Chinese/Japanese support. citeturn0search0turn0search3
- EasyOCR: multilingual OCR including Simplified/Traditional Chinese, Japanese and Latin. citeturn0search7
- Kraken: historical/non-Latin OCR adapter, already supported by this project.
- Tesseract: general OCR adapter where suitable trained data exists.

## Data exchange and SQL

The project provides SQLite plus portability templates for PostgreSQL and MySQL. JSON and CSV are first-class flat-file exchange formats. Microsoft Access is supported through an optional `pyodbc` adapter and an installed Access ODBC driver; no proprietary driver is bundled.

Additional database adapters should preserve the canonical schema semantics: provenance, source hashes, OCR confidence, translation mode, alternatives and review state.

## Licensing/provenance

External projects are research and architecture references. Their source code, models, weights and data remain subject to their own licenses and terms. This repository implements its own integration contracts and does not silently redistribute third-party proprietary assets.
