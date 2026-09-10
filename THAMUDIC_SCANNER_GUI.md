# Ancient Script Scanner — Python GUI + Web UI

The Thamudic research system is now Python-first across both desktop and web interfaces.

## Run

```bash
python3 -m pip install -r requirements-thamudic.txt
python3 run_thamudic.py desktop
python3 run_thamudic.py web --host 127.0.0.1 --port 5000
```

## Professional window hierarchy

Both interfaces share the same information architecture:

1. Dashboard
2. Scanner
3. Translator
4. Historical Objects
5. Inscriptions
6. Ancient Scripts
7. Sources & Rights
8. Database
9. Research

The desktop implementation is `thamudic_desktop.py` using Tkinter/ttk. `thamudic_scanner_gui.py` remains a compatibility launcher. The web implementation is `thamudic_web_app.py` using Flask.

## Main functionality

1. Import PNG/JPEG/TIFF/BMP/WebP images or render the first page of a PDF.
2. Select script/variety and broad historical period.
3. Run normalization and connected-component segmentation.
4. Review transliteration and Arabic/English interpretation separately.
5. Record provenance and bibliography.
6. Add the evidence record to the local SQLite historical-object database.
7. Search/filter catalog records by period, script, object type, country and text through the shared database API.
8. Export stable Softr CSV/JSON schemas.
9. View source institutions and image-rights policies.
10. Use the same evidence records from the desktop and Flask web application.

## Database

`ancient_objects_db.py` stores objects, annotations and source metadata. `historical_periods.py` covers Paleolithic, Epipaleolithic, Neolithic, Chalcolithic, Bronze Age, Iron Age, Hellenistic/Greek, Roman, Byzantine/Eastern Roman, Early Islamic, Medieval, Early Modern and Modern labels.

## External object/image sources

The catalog architecture is prepared for OCIANA, DASI, The Metropolitan Museum of Art Open Access, Smithsonian Open Access, Europeana and IIIF. The application stores image URLs/IIIF references and rights statements instead of assuming that every remotely visible image can be republished.

## Softr / hosted-site migration

`softr_export.py` produces a CSV/JSON import shape. `softr_api.py` provides optional REST synchronization using an environment variable (`SOFTR_API_KEY`) and database/table identifiers. No credential is stored in GitHub.

The Bubble, BuiltWithRocket and Softr deployments are treated as functional/reference systems. Their platform internals are not copied. The owned replacement is implemented in Python and backed by the canonical SQLite database.

## Research principle

The scanner distinguishes **segmentation**, **recognition candidates**, **transliteration**, and **translation**. It never presents an OCR candidate as an authoritative scholarly reading. Competing readings, provenance, source record IDs and image rights remain explicit fields.
