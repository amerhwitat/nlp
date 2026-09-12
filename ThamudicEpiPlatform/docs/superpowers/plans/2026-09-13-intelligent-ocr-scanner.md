# Intelligent OCR Scanner Implementation Plan

## Goal

Add a provenance-aware, pluggable OCR scanner to the canonical Thamudic/Ancient Languages platform and expose the same contract to the `general` integration repository and all maintained language clients.

## Design

1. **Bounded input** — reject empty/oversized image payloads and retain SHA-256 identity.
2. **Image intelligence** — calculate resolution, mean intensity, contrast and blur indicators before recognition.
3. **Engine adapters** — support `auto`, `kraken`, `tesseract` and `quality-only`. Kraken is optional and model-driven; Tesseract is optional and depends on installed traineddata.
4. **Confidence routing** — in `auto`, select the highest-confidence successful engine result and preserve engine identity.
5. **Unicode/script routing** — normalize recognized text to NFC and score candidate ancient scripts from Unicode ranges, including Old North Arabian, Old South Arabian, Nabataean, Phoenician, Imperial Aramaic, Cuneiform, Egyptian Hieroglyphs, Coptic and Linear B.
6. **Evidence preservation** — return boxes, confidence, warnings, preprocessing steps and source hash. OCR remains recognition-only and cannot silently become a transliteration or translation.
7. **Persistence/KPIs** — record OCR jobs, confidence, warnings and engine in SQLite and expose job/error/confidence KPIs.
8. **UI** — add image scanning to the research UI and display OCR output, script candidates, confidence and warnings next to application KPIs.
9. **Cross-language clients** — Python, C++, C#, Java, Go, Rust, JavaScript and TypeScript clients invoke the same HTTP contract.
10. **Automation** — Bash, PowerShell and CMD dependency checks install Python requirements and optionally Kraken; language build scripts compile available clients; deployment scripts invoke both dependency and language-build checks.
11. **Documentation/citations** — cite Unicode, Kraken, Tesseract and cuneiform OCR research and preserve licensing boundaries.

## Acceptance criteria

- `/api/ocr/scan` returns a stable, schema-backed recognition result.
- OCR failures produce explicit warnings rather than fabricated text.
- OCR jobs are represented in KPI output.
- The UI can upload an image and display the result.
- Every maintained language client has an OCR invocation path or common HTTP implementation.
- Bash/PowerShell/CMD automation checks and installs dependencies before build/deploy.
- General-repository integration exposes the same OCR contract without becoming a second source of truth.
- Documentation identifies third-party engines/models as external references and does not copy their proprietary/model assets.
- Tests cover scanner provenance, size limits and PDF/legacy API compatibility.
