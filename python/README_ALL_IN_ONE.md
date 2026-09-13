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
- Optional modern-language desktop TTS
- GUI controls to read transliteration and translated text aloud
- Voice stop control and adjustable speech rate
- Voice-enabled launcher: `ThamudicScanner_AllInOne_Voice.py`
- Tkinter desktop GUI
- CLI operation

## Run

Standard all-in-one GUI:

```bash
python python/ThamudicScanner_AllInOne.py --gui
```

Voice-enabled GUI:

```bash
python python/ThamudicScanner_AllInOne_Voice.py --gui
```

Install the optional desktop TTS engine when needed:

```bash
pip install pyttsx3
```

The voice layer prefers an installed operating-system voice matching the requested language (English for scholarly transliteration and the selected target language for translations). It falls back to the system default voice when language metadata is unavailable.

## Modular compatibility

The original modules remain in `python/thamudic/` for library users, tests, API/server integration, and maintainability. The all-in-one file does not delete or replace those modules. When repository data files are present, the all-in-one implementation loads the same registries; when they are absent, it uses a conservative embedded fallback for core scripts.

## Research safety

Unicode script recognition is not proof of language identification. Corpus-backed translations are distinguished from unsupported readings. The application does not invent a native pronunciation for ancient languages. Unknown translations are explicitly reported as unavailable/provider-required. TTS is a playback convenience for transliteration/modern-language translations, not evidence that an ancient script has a historically established native pronunciation.
