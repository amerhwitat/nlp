# Thamudic / Ancient North Arabian Scanner Research

## Scope

This module is a research scanner for photographs and scans of Ancient North Arabian (ANA) inscriptions, including material traditionally grouped as Thamudic. It is a segmentation and evidence-preservation layer, not an automatic scholarly translator.

## Important terminology

OCIANA explains that “Thamudic” is a historical umbrella term for a large set of Ancient North Arabian inscriptions. Thamudic A was later identified as Taymanitic, Thamudic E as Hismaic, and Thamudic F as Himaitic; B, C and D remain less fully characterized. Therefore every record should preserve script/variety, corpus identifier, location, date and confidence rather than flattening all ANA material into one alphabet.

## Current research targets

- OCIANA Thamudic overview and searchable corpus: https://ociana.osu.edu/scripts_thamudic
- OCIANA example Thamudic D inscription RZTham 1: https://ociana.osu.edu/inscriptions/44213
- OCIANA example Thamudic D inscription TIJ 112.1: https://ociana.osu.edu/inscriptions/7258
- OCIANA example Thamudic B inscription GETham 2: https://ociana.osu.edu/inscriptions/44105
- OCIANA Safaitic example TIJ 503: https://ociana.osu.edu/inscriptions/2400
- Unicode Old North Arabian chart: https://www.unicode.org/charts/nameslist/n_10A80.html
- Unicode Core Specification, Old North Arabian: https://www.unicode.org/versions/Unicode16.0.0/core-spec/chapter-10/
- Writing Systems Technical Resources — Old North Arabian: https://writingsystems.info/scrlang/scripts/narb/
- Wikimedia Commons Thamudic/Jubbah photographic examples: https://commons.wikimedia.org/wiki/File:Thamudic_inscription_at_Jubbah_Rock_Art_Site_(1).jpg

## User-supplied recent web projects

These are recorded as external reference/demo targets. They are not treated as source code unless their owners provide an open-source repository or license.

1. Bubble prototype: https://thamudicscan.bubbleapps.io/version-test
2. BuiltWithRocket translator UI: https://thamudicscan-s3wz30.public.builtwithrocket.new/
3. Softr historical artifacts UI: https://thamudic-scanner.softr.app/

## Open-source engineering references

The following projects provide transferable architecture ideas rather than Thamudic-specific code:

- CuReD / DigitalPasts: https://github.com/DigitalPasts/CuReD — human-in-the-loop OCR for cuneiform transliterations, with data/models/tests and reproducible training workflow.
- Coptic Scriptorium OCR: https://github.com/CopticScriptorium/OCR — open development workflow for historical-script OCR and ground truth.
- MAS: https://github.com/ai-forever/MAS — historical Arabic manuscript OCR benchmark and training configurations.
- Arabic-OCR examples: https://github.com/ayk1993/Arabic-OCR and https://github.com/maidaly/Arabic_OCR — modern Arabic OCR engineering references.
- Hieroglyphic OCR: https://github.com/nederhof/hocr — experimental image-to-Unicode transcription architecture for an ancient script.

These projects demonstrate useful patterns: image preprocessing, segmentation, ground-truth datasets, model/version separation, human review, reproducible evaluation and Unicode output. Their code and data must remain subject to their individual licenses; this repository does not copy third-party implementations.

## Unicode model

The Unicode Old North Arabian block is U+10A80–U+10A9F. Unicode documents 28 letter code points plus three number code points. Directionality and glyph mirroring can vary by inscription/line; therefore the scanner stores image geometry and does not assume that Unicode rendering orientation alone identifies the physical carving.

## Scanner architecture

1. Ingest photograph/scan.
2. EXIF-aware orientation and grayscale normalization.
3. Contrast normalization and deterministic thresholding.
4. Connected-component candidate extraction.
5. Bounding-box and shape statistics.
6. Preserve original image and provenance metadata.
7. Optional future classifier plug-in for glyph recognition.
8. Human verification and corpus matching.
9. Export JSON/CSV evidence records.
10. Optional web/API layer can consume the same schema.

The baseline implementation intentionally uses Pillow + NumPy and does not require Tesseract or camel_tools. Recognition is separated from segmentation because Ancient North Arabian varieties contain local/epigraphic variation and because a segmentation box alone is not evidence for a transliteration.

## Proposed data record

Each inscription record should contain:

- source image and checksum
- photographer/source/license
- corpus and inscription identifier
- geographic coordinates when legitimately published
- site and current location
- script variety: Dadanitic, Safaitic, Hismaic, Taymanitic, Thamudic B/C/D, etc.
- direction and line geometry
- glyph bounding boxes
- Unicode candidate(s)
- transliteration candidate(s)
- Arabic/Hebrew/Latin display values where useful
- model version and training dataset
- segmentation confidence
- recognition confidence
- scholarly verification status
- notes/apparatus and competing readings

## Research safety and scholarly quality

The scanner must distinguish visual evidence from linguistic inference. OCR/HTR output is a candidate reading, not an authoritative translation. “Pictographic meaning” should never be inferred merely from glyph shape. Corpus identifiers and published scholarly readings should be preserved as provenance.

## Future roadmap

- Build a labelled glyph dataset from legitimately reusable photographs and facsimiles.
- Add line detection and curved/rotated inscription handling.
- Add augmentation for erosion, shadows, cracks and uneven illumination.
- Train a small CPU-safe classifier for glyph candidates.
- Add KMeans/PCA exploratory clustering for variant discovery.
- Add optional TensorFlow/Keras model loading without changing the ingestion schema.
- Add OCIANA record linking and evidence comparison.
- Add Arabic/English/Hebrew UI and transliteration editor.
- Add artifact database records and web API.
- Benchmark against human-reviewed ground truth using character error rate and per-glyph confusion matrices.
