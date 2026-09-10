# Jordan Heritage Dataset

This seed catalog is designed for the Thamudic / Ancient North Arabian scanner and historical-object database.

## Coverage

The initial reviewed seed covers major Jordanian archaeological contexts across the Neolithic, Iron Age, Hellenistic, Roman, Byzantine and Early Islamic periods, including Ain Ghazal, Amman, Petra, Dhiban, Gadara/Umm Qais, Madaba, Quseir Amra, Umm er-Rasas, Jerash and Umm Al-Jimal.

The Department of Antiquities of Jordan maintains a databases portal pointing researchers to OCIANA, Manar al-Athar, Nabataean Studies, Department publications, Islamic inscriptions and Qasr al-Mushatta resources.

## Image policy

Images are referenced rather than indiscriminately copied into the repository. Each record stores:

- source URL
- image page URL
- image URL when an appropriate reusable image reference was identified
- creator
- license / rights note
- provenance

A Commons image is not automatically treated as public domain. The item-level license must be followed. Museum and UNESCO media may have separate rights from their descriptive text.

## Scholarly policy

- Site chronology is broad unless a source provides a more precise date.
- Script and language are separate fields.
- OCR/recognition is not equivalent to translation.
- Competing readings should remain possible.
- Source identifiers and bibliography should be preserved.
- Archaeological objects must not be represented as authenticated when the source only provides a photograph or secondary description.

## Softr population

The seed can be reviewed first:

```bash
python scripts/populate_softr_jordan.py --database-id DB_ID --table-id TABLE_ID --dry-run
```

To populate a connected Softr Database:

```bash
export SOFTR_API_KEY='...'
python scripts/populate_softr_jordan.py --database-id DB_ID --table-id TABLE_ID
```

No Softr credential is stored in GitHub.
