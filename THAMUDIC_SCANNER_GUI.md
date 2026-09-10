# Ancient Script Scanner — Python GUI

`thamudic_scanner_gui.py` is the primary Tkinter window for the Thamudic / Ancient North Arabian research scanner **and historical-object catalog**.

## Run

```bash
python3 -m pip install -r requirements-thamudic.txt
python3 thamudic_scanner_gui.py
```

## Main functionality

1. Import PNG/JPEG/TIFF/BMP/WebP images or render the first page of a PDF.
2. Select script/variety and broad historical period.
3. Run normalization and connected-component segmentation.
4. Review transliteration and Arabic/English interpretation separately.
5. Record provenance, bibliography and reviewer confidence.
6. Add the evidence record to the local SQLite historical-object database.
7. Search/filter catalog records by period and text.
8. Export a stable Softr CSV/JSON schema.
9. View source institutions and image-rights policies.

## Database

`ancient_objects_db.py` stores objects, annotations and source metadata. `historical_periods.py` covers Paleolithic, Epipaleolithic, Neolithic, Chalcolithic, Bronze Age, Iron Age, Hellenistic/Greek, Roman, Byzantine/Eastern Roman, Early Islamic, Medieval, Early Modern and Modern labels.

## External object/image sources

The catalog architecture is prepared for OCIANA, DASI, The Metropolitan Museum of Art Open Access, Smithsonian Open Access, Europeana and IIIF. The application stores image URLs/IIIF references and rights statements instead of assuming that every remotely visible image can be republished.

## Softr

`softr_export.py` produces a CSV/JSON import shape. `softr_api.py` provides optional REST synchronization using an environment variable (`SOFTR_API_KEY`) and database/table identifiers. No credential is stored in GitHub.

The published Softr app itself could not be fetched by the available crawler during the current update, so private app/database records were not falsely represented as copied. See `SOFTR_DATABASE_MIGRATION.md` for the supported migration/API workflow.

## Research principle

The scanner distinguishes **segmentation**, **recognition candidates**, **transliteration**, and **translation**. It never presents an OCR candidate as an authoritative scholarly reading. Competing readings, provenance, source record IDs and image rights remain explicit fields.
