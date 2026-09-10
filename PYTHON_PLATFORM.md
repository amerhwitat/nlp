# Thamudic Python Platform

The three historical web experiences are now represented by one owned Python application layer rather than a mixture of platform-specific front ends.

## Migration policy

The Bubble application, BuiltWithRocket application, and Softr catalog are treated as functional references and data sources. Their proprietary platform internals are not copied. Bubble currently does not provide traditional editable source-code export; migration therefore means reimplementing the user-visible behavior in owned Python code. See Bubble's current documentation for its export limitations.

## Python-first architecture

```text
Image / PDF
    |
    v
thamudic_scanner.py
    |
    +--> glyph segmentation / evidence
    |
    v
ancient_objects_db.py  <--> ancient_objects.sqlite
    |
    +--> thamudic_desktop.py       (Tkinter/ttk desktop)
    |
    +--> thamudic_web_app.py       (Flask web UI/API)
    |
    +--> softr_export.py           (CSV/JSON interoperability)
    |
    v
Chimera II OS / Aurora integration
```

## Run

Desktop:

```bash
python run_thamudic.py desktop --db ancient_objects.sqlite
```

Web:

```bash
python run_thamudic.py web --db ancient_objects.sqlite --host 127.0.0.1 --port 5000
```

The web application exposes Dashboard, Scanner, Translator, Historical Objects, Inscriptions, Ancient Scripts, Sources & Rights, Database, and Research views. JSON API endpoints provide object search and record retrieval.

## Design principles

- Python is the application language for both desktop and web layers.
- SQLite remains the canonical persistent evidence store.
- JSON, CSV and SQL remain interchange/export formats.
- Scanner segmentation is explicitly separated from scholarly recognition/translation.
- Source provenance and rights remain attached to records.
- Desktop and web applications use the same database and information hierarchy.
- No Tesseract or camel_tools dependency is introduced by the scanner baseline.

## Relationship to the three existing sites

| Existing experience | Python replacement |
|---|---|
| Thamudic Scanner | `/scanner` + `ThamudicDesktop` Scanner workspace |
| Thamudic Translator | `/translator` + desktop Translator workspace |
| Historical Objects / Artifacts | `/artifacts` + desktop Historical Objects workspace |
| Softr data export | Existing `softr_export.py` + SQLite/JSON/CSV |

This repository contains the implementation; external hosted applications remain useful as historical/reference deployments until their data and behavior are fully migrated.
