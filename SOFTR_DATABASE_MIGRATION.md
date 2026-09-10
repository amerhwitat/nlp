# Softr Database Migration

The public app `thamudic-scanner.softr.app` could not be fetched by the available web crawler in this session, so its private Softr database/schema cannot be copied or reverse-engineered from the published page alone. The Python project now provides a clean migration path instead.

## 1. Export

Run the Tkinter app and choose **Export Softr CSV**, or use:

```python
from ancient_objects_db import ObjectDatabase
from softr_export import export_softr_csv

db = ObjectDatabase('ancient_objects.sqlite')
export_softr_csv(db.list_objects(), 'softr_objects.csv')
```

## 2. Import

Softr supports CSV import into Softr Databases. Map the exported columns to the desired fields and keep `Record ID` as the stable external identifier.

## 3. Live API synchronization

The optional `softr_api.py` client uses `SOFTR_API_KEY`. The API also needs the target database and table IDs. Example setup:

```text
Windows PowerShell:
$env:SOFTR_API_KEY = '...'

Linux/macOS:
export SOFTR_API_KEY='...'
```

Never commit the token. The client supports listing databases/tables, reading records, creating records and updating records.

## 4. Recommended Softr tables

- **Objects** — title, period, object type, culture, script, dates, site, material, description, source, image, rights, provenance.
- **Annotations** — object ID, bounding box, glyph candidate, transliteration candidate, confidence, reviewer.
- **Sources** — institution, homepage, API/record URL, rights policy, image policy.
- **Periods** — period key, name, parent, start/end, notes.
- **Readings** — object ID, reading type, transliteration, Arabic/English interpretation, reviewer, confidence, status.

This design allows the Web UI to evolve without collapsing scholarly evidence, archaeological metadata and image rights into one field.
