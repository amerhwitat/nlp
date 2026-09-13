# Thamudic Scanner — All-in-One Python Implementation

`ThamudicScanner_AllInOne.py` is the single-file runnable implementation of the Python scanner. It consolidates the runtime responsibilities of the modular `python/thamudic` package into one copyable entry point.

## Included

- Old North Arabian / Thamudic Unicode recognition and UTF-8 inspection
- Transliteration using the repository's Old North Arabian mapping
- Evidence-aware Ancient North Arabian corpus translation
- Ancient-script registry and historical metadata loading
- Translation capability/direction reporting
- Append-only translation history with SHA-256 record integrity checks
- Translation-history PDF generation
- Script-report data generation
- Optional modern-language TTS capability detection
- Tkinter desktop GUI
- CLI operation

## Run

```bash
python python/ThamudicScanner_AllInOne.py --gui
```

CLI examples:

```bash
python python/ThamudicScanner_AllInOne.py --text "..." --scan --script ancient-north-arabian
python python/ThamudicScanner_AllInOne.py --text "..." --translate --script ancient-north-arabian --target en
python python/ThamudicScanner_AllInOne.py --history-pdf translation-history.pdf
python python/ThamudicScanner_AllInOne.py --verify-history
```

PDF output requires the project's ReportLab dependency. GUI operation requires Tkinter, which is normally supplied by the Python distribution on desktop systems.

## Modular compatibility

The original modules remain in `python/thamudic/` for library users, tests, API/server integration, and maintainability. The all-in-one file does not delete or replace those modules. When repository data files are present, the all-in-one implementation loads the same registries; when they are absent, it uses a conservative embedded fallback for core scripts.

## Research safety

Unicode script recognition is not proof of language identification. Corpus-backed translations are distinguished from unsupported readings. The application does not invent a native pronunciation for ancient languages. Unknown translations are explicitly reported as unavailable/provider-required.
