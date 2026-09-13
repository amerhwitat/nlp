# Thamudic All-in-One media import safety

## Windows OpenMP / `libiomp5md.dll`

The All-in-One scanners initialize `python/thamudic/runtime.py` before scientific/OCR modules. On Windows, the compatibility fallback sets `KMP_DUPLICATE_LIB_OK=TRUE` unless `THAMUDIC_ALLOW_DUPLICATE_OPENMP=0` is explicitly set.

The preferred architecture is now stronger than an environment-variable workaround: EasyOCR/PyTorch runs in `python/thamudic/ocr_worker.py`, a child process. This means a native DLL/OpenMP initialization failure in OCR cannot terminate the Tkinter GUI process.

## Image import

1. Select **Import image / PDF**.
2. The parent GUI stays alive while OCR runs in the worker.
3. OCR results and confidence are returned as JSON.
4. If OCR cannot initialize, the GUI receives an `ocr_error` result rather than exiting.

Useful environment variables:

- `THAMUDIC_OCR_TIMEOUT` — OCR worker timeout in seconds; default `180`.
- `THAMUDIC_ALLOW_DUPLICATE_OPENMP=0` — disable the Windows compatibility fallback when the installed native stack has been cleaned up to one OpenMP runtime.

## PDF import

Text PDFs use `pypdf` directly. Scanned PDFs render each page through the isolated worker using `pypdfium2`, then OCR the rendered page through EasyOCR. Individual page failures are recorded in `ocr_errors` and do not terminate the application.

## Evidence policy

An OCR failure never becomes a guessed inscription. Empty/failed OCR remains unavailable and is reported in metadata. Translation is attempted only when extracted text exists and follows the project's evidence-backed translation policy.

## Regression coverage

`python/tests/test_media_pipeline_safety.py` verifies normal text import and that an OCR exception becomes a non-fatal metadata result. CI also compiles the all-in-one entry points and the Thamudic package.
