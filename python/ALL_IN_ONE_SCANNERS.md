# All-in-One Thamudic and NLP Scanners

The repository now provides two self-contained Python desktop applications:

- `python/ThamudicScanner_AllInOne.py`
- `python/NLPScanner_AllInOne.py`

Each file contains its own runtime safeguards, Unicode/script scanning, Old North
Arabian transliteration, evidence-backed sample translation records, media import,
OCR/PDF worker mode, PDF export, voice support, CLI and Tkinter GUI.

## No local package dependency

The two entry points do not require importing `thamudic.*` modules. This makes them
portable as standalone applications and prevents a missing local package import from
breaking startup.

## Crash-safe image/PDF handling

The application launches itself as a child worker for EasyOCR/PyTorch and pypdfium2.
The GUI process therefore remains alive if a native DLL, OpenMP runtime, OCR model,
or PDF renderer fails. Timeouts and structured JSON errors are used at the process
boundary.

Text PDFs use `pypdf` first. Scanned PDFs are rendered page-by-page and sent through
the isolated OCR worker. OCR failures are reported as diagnostics; they are never
converted into fabricated historical readings.

## Runtime controls

Windows compatibility can be disabled for diagnosis with:

`THAMUDIC_ALLOW_DUPLICATE_OPENMP=0`

OCR/PDF worker limits can be changed with `THAMUDIC_OCR_TIMEOUT` and
`THAMUDIC_PDF_RENDER_TIMEOUT`. The NLP application accepts the corresponding
`NLP_OCR_TIMEOUT`, `NLP_PDF_RENDER_TIMEOUT`, and `NLP_ALLOW_DUPLICATE_OPENMP` variables.

## Examples

```text
python python/ThamudicScanner_AllInOne.py --help
python python/ThamudicScanner_AllInOne.py inscription.pdf
python python/ThamudicScanner_AllInOne.py --text "..." --scan
python python/ThamudicScanner_AllInOne.py --text "..." --translate --target en

python python/NLPScanner_AllInOne.py --help
python python/NLPScanner_AllInOne.py inscription.jpg
python python/NLPScanner_AllInOne.py --text "..." --scan
```

The included GitHub Actions workflow performs syntax compilation, CLI help checks,
and dependency-free text scanning. OCR/PDF tests remain environment-dependent and
are intentionally not run in the lightweight smoke job.
