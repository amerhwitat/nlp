# Ancient Script Scanner — Python GUI + Web UI

The Thamudic research system is Python-first across desktop and web interfaces and now has a shared integration path into `AncientVisualResearchSuite/`.

## Run

```bash
python3 -m pip install -r requirements-thamudic.txt
python3 run_thamudic.py desktop
python3 run_thamudic.py web --host 127.0.0.1 --port 5000
```

## Professional window hierarchy

1. Dashboard
2. Scanner
3. Translator
4. Historical Objects
5. Inscriptions
6. Ancient Scripts
7. Sources & Rights
8. Database
9. Research
10. Historical Event Visualization

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
11. Send selected evidence into `AncientVisualResearchSuite/` to construct a time/location-bound historical scene.
12. Synchronize the scene timeline with excavation findings, artifact positions, environmental layers, characters and astronomical context.

## Historical visualization integration

`AncientVisualResearchSuite/` provides the next stage after OCR and cataloging:

`inscription image -> glyph evidence -> inscription/site -> date/region -> historical event -> map -> environment -> sky -> 3D artifact -> evidence-linked visualization`

Recognition candidates remain distinct from scholarly readings. Reconstructed characters, buildings and events carry evidence IDs and confidence classes.

## Database

`ancient_objects_db.py` stores objects, annotations and source metadata. `historical_periods.py` covers Paleolithic, Epipaleolithic, Neolithic, Chalcolithic, Bronze Age, Iron Age, Hellenistic/Greek, Roman, Byzantine/Eastern Roman, Early Islamic, Medieval, Early Modern and Modern labels.

## Research principle

The scanner distinguishes **segmentation**, **recognition candidates**, **transliteration**, and **translation**. It never presents an OCR candidate as an authoritative scholarly reading. Competing readings, provenance, source record IDs and image rights remain explicit fields. Historical reconstructions are likewise labeled as observed, supported, inferred, speculative or visualization-only.
