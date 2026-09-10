# Ancient Script Scanner — Python GUI

The main desktop interface is `thamudic_scanner_gui.py` and uses **Tkinter**. It is the primary import/export window for the Thamudic / Ancient North Arabian research scanner and the broader ancient-script registry.

## Run

```bash
python3 -m pip install pillow numpy
python3 thamudic_scanner_gui.py
```

Tkinter is included with most standard Python installations. On some Linux distributions it is supplied by the `python3-tk` system package.

## Workflow

1. **Import Image** — PNG/JPEG/TIFF/BMP/WebP inscription photograph.
2. **Select script/variety** — Old North Arabian, Thamudic B/C/D, Taymanitic, Hismaic, Himaitic, Safaitic, Dadanitic, Dumaitic, Hasaitic, Ancient South Arabian varieties, Phoenician, Aramaic, Nabataean.
3. **Scan** — normalization and connected-component segmentation.
4. **Review** — enter or correct transliteration, Arabic notes/translation, English notes/translation and provenance.
5. **Export JSON/CSV/TXT** — retain machine evidence and human scholarly notes together.

## Design principle

The scanner distinguishes **image segmentation**, **recognition candidates**, **transliteration**, and **translation**. It does not present an OCR candidate as an authoritative scholarly translation. Competing readings and provenance should be retained.

## Related sources

- OCIANA: https://ociana.osu.edu/
- Unicode Old North Arabian: https://www.unicode.org/charts/nameslist/n_10A80.html
- User-provided Bubble prototype: https://thamudicscan.bubbleapps.io/version-test
- User-provided translator UI: https://thamudicscan-s3wz30.public.builtwithrocket.new/
- User-provided artifact database UI: https://thamudic-scanner.softr.app/

External applications are reference/demo targets only. Their source code and data are not copied into this repository without permission and license review.
