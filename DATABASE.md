# Ancient Object Research Database

## Decision

The project uses **SQLite as the canonical database**. Flat JSON remains the portable interchange format, and CSV/SQL dumps are supported for migration and archival. Microsoft Access is not required and is intentionally not the primary format because the Python application needs a cross-platform, dependency-light database.

SQLite is a single-file, serverless, zero-configuration SQL database and Python exposes it through `sqlite3`. See the official documentation: https://sqlite.org/ and https://docs.python.org/3/library/sqlite3.html

## Files

- `ancient_objects_db.py` — database API used by the GUI and other Python tools.
- `database_cli.py` — command-line management utility.
- `data/ancient_objects_schema.sql` — portable schema.
- `data/*.json` — reviewed seed/interchange data.
- `ancient_objects.sqlite` — generated runtime database; normally created locally and not committed as source code.

## Database model

### `objects`

Stores archaeological objects, inscriptions, artifacts, sites and evidence metadata: period, culture, script, language, dates, site, region, country, material, description, transliteration, translations, sources, image/IIIF links, rights, provenance, bibliography, coordinates, confidence, review status and tags.

### `annotations`

Stores image regions/glyph candidates linked to an object, including bounding-box coordinates, Unicode candidates, transliteration candidates, reviewer confidence and notes.

### `sources`

Stores normalized source providers and their API/rights/image policies.

### `database_meta`

Stores schema and storage-format metadata for forward migrations.

## Usage

Initialize:

```bash
python database_cli.py --db data/ancient_objects.sqlite init
```

Import JSON:

```bash
python database_cli.py --db data/ancient_objects.sqlite import-json data/jordan_heritage_seed.json
python database_cli.py --db data/ancient_objects.sqlite import-json data/jordan_high_value_artifacts_seed.json
python database_cli.py --db data/ancient_objects.sqlite import-json data/jordan_epigraphy_seed.json
```

Search:

```bash
python database_cli.py --db data/ancient_objects.sqlite search Safaitic --country Jordan
```

Export JSON/CSV/SQL:

```bash
python database_cli.py --db data/ancient_objects.sqlite export-json backup.json
python database_cli.py --db data/ancient_objects.sqlite export-csv backup.csv
python database_cli.py --db data/ancient_objects.sqlite export-sql backup.sql
```

Make a consistent binary SQLite backup:

```bash
python database_cli.py --db data/ancient_objects.sqlite backup backups/ancient_objects.sqlite
```

## Integration

`thamudic_scanner_gui.py` already opens `ObjectDatabase`, writes reviewed scan records, lists/searches catalog records and exports Softr data. The database layer is deliberately independent of Tkinter so the same records can be used by a web UI, batch ingestion, OCR pipelines and future Chimera II OS services.

## Data policy

Remote images remain URL/IIIF references by default. The database stores source, page, creator, license and rights notes rather than assuming that a web image is reusable. Scholarly translations and machine/OCR candidates are kept distinct from reviewed evidence.
