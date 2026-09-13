# Unified Thamudic + NLP scanner fix

The desktop implementation is now consolidated in `thamudic_all_in_one.py`. The NLP entry point remains a compatibility launcher, so `run_thamudic.py desktop` and `run_thamudic.py nlp` use the same GUI hierarchy and the same scanner engine.

## GUI parity

The unified window contains the same research-workbench shell and navigation used by the main Thamudic desktop application:

- Dashboard
- Scanner
- Translator
- Historical Objects
- Inscriptions
- Ancient Scripts
- Sources & Rights
- Database
- Research

The action area also exposes Import Image / PDF, Scan Current Image, NLP Analyze, Translate / Transliterate, Export Report / JSON, Print Report, Clear Workspace, and Add reviewed evidence.

## Local worker runtime fix

Image processing is no longer executed directly inside the Tkinter event handler. A bounded `ThreadPoolExecutor` performs the CPU/image work in a local worker. The worker returns a structured result through a thread-safe `queue.Queue`; only the Tkinter main thread updates widgets via a short `after()` polling loop.

This prevents worker exceptions from escaping into the GUI callback and prevents Tkinter widgets from being touched by the worker thread. Each scan has a generation token, so a stale result cannot overwrite a newer scan or a cleared workspace.

`scan_image_safe()` is also a hard exception boundary and returns `ok`, `error_type`, `error`, and diagnostic traceback information instead of raising across the worker boundary.

## Output pipeline

- Transliteration, Arabic translation and English translation are explicitly populated after analysis.
- GUI rebuilding no longer loses output.
- Translator and NLP actions write all three visible output fields.
- Unicode Old North Arabian (`U+10A80–U+10A9C`) uses the embedded deterministic mapping.
- The offline lexical layer provides candidate glosses for common scholarly transliterations.
- PDF first-page import is supported through PyMuPDF.
- JSON, CSV, text and PDF research reports can be exported and the generated report can be sent to the system printer.
- Tesseract and camel_tools are not used.

## Recognition boundary

Image segmentation is visual evidence, not a linguistic reading. The application therefore does not fabricate an inscription from bounding boxes. A validated glyph-recognition model is required for automatic image-to-glyph reading. Once a model is available, its reviewed output can enter the same visible source-text, transliteration and translation pipeline without changing the GUI.
