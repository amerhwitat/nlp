# Thamudic output pipeline fix

The launcher now defaults to `thamudic_all_in_one.py` and supports `nlp` mode through `nlp_thamudic_all_in_one.py`.

## Fixed behavior

- Transliteration, Arabic translation and English translation are explicitly populated after analysis.
- GUI rebuilding no longer loses output: results are computed first, the scanner view is rebuilt, and `_populate_result()` writes the result into the newly created text widgets.
- The Translator page writes all three output fields directly.
- NLP analysis writes the same fields plus token/evidence information.
- Unicode Old North Arabian input (`U+10A80–U+10A9C`) is transliterated through the embedded mapping.
- A small offline lexical layer translates common scholarly transliterations such as `mlk`, `bn`, `byt`, `ʿbd`, `ʾrs`, `ywm`, and `šnt` into Arabic and English candidate glosses.
- PDF first-page import remains supported through PyMuPDF.
- Tesseract and camel_tools are not used.

## Important recognition boundary

Image segmentation does not by itself determine the linguistic identity of a glyph. Therefore the application must not invent a transliteration from a bounding box. Automatic image-to-glyph reading requires a trained, validated glyph classifier. The all-in-one engine exposes the correct output pipeline and can accept a future model without changing the GUI/result schema.

For an image-only scan with no recognizer, the UI reports candidate components and leaves the linguistic fields ready for a Unicode ONA or scholarly transliteration to be entered/verified.
