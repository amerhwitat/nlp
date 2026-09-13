# Desktop Voice Controls

The Thamudic and general NLP Tkinter scanners can read generated transliterations and translations aloud.

## Controls

- **Read transliteration** — sends the scholarly transliteration to desktop TTS using an English voice preference.
- **Read translation** — sends the translated text to desktop TTS using the selected target language (`en` or `ar`) when a matching installed voice can be identified.
- **Stop** — stops the active desktop speech engine.
- **Rate** — adjusts speech speed from 60–300 words/minute.

## Backends

The GUI uses the optional `pyttsx3` desktop TTS backend. If it is not installed, the GUI reports that TTS is unavailable rather than failing the scanner.

```bash
pip install pyttsx3
```

The shared implementation is `python/thamudic/desktop_voice.py`.

## Ancient-language pronunciation policy

The application does **not** claim that an operating-system voice is a historically authentic Thamudic, Safaitic, Dadanitic, Hismaic, or other ancient-language voice. Transliteration is read as a modern-language pronunciation aid. Native ancient-script pronunciation remains provider-gated unless a documented scholarly pronunciation resource is supplied.

## All-in-one launcher

For the consolidated scanner plus voice controls:

```bash
python python/ThamudicScanner_AllInOne_Voice.py --gui
```

The general NLP all-in-one GUI includes the same controls directly:

```bash
python python/NLPScanner_AllInOne.py --gui
```
