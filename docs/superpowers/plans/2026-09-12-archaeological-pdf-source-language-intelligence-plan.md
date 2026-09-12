# Archaeological PDF & Source-Language Intelligence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add secure PDF reading/generation and evidence-aware source-language reasoning so every archaeological object's translation presents original source text, scholarly transliteration, English transliteration, Arabic transliteration, and the selected target-language translation with explicit reasoning and uncertainty.

**Architecture:** Build a shared document contract and service layer under `ArchaeologicalKnowledgeSystem`, with Python as the reference OCR/PDF processing implementation and language-neutral JSON contracts consumed by TypeScript/Node and PHP. Keep PDF extraction, page rendering, OCR, transliteration, semantic reasoning, and PDF export as separate adapters so engines can be replaced without changing archaeological records. Store every derived result as provenance-bearing observations rather than overwriting source evidence.

**Tech Stack:** Python 3.10+ reference services, TypeScript/Node.js web services, PHP API adapter, JSON Schema, SQLite/PostgreSQL-compatible SQL, reportlab for generated PDFs, PyMuPDF/PDFium-compatible adapter boundary for reading/rendering, existing Thamudic scanner adapters, Unicode NFC/UTF-8, BCP-47, Web Workers/PWA, pytest and contract fixtures.

**Spec:** `docs/superpowers/specs/2026-09-12-archaeological-language-intelligence-design.md`

## Global Constraints

- Application code remains isolated under `ArchaeologicalKnowledgeSystem/`.
- Existing `data/ancient_north_arabian/alphabet.json` remains the authoritative Ancient North Arabian registry.
- Source text is immutable evidence; OCR, transliteration, translation, and reasoning are derived records.
- Unicode/UTF-8/NFC is canonical for stored text; preserve original code points and byte representation when available.
- Never imply that Unicode support means reliable OCR or translation support.
- Every inferred reading or translation records confidence, engine/model/version, timestamp, and evidence references.
- PDF/image ingestion must preserve page, region, source, rights, and checksum provenance.
- Rights metadata is retained; copyrighted media is not blindly downloaded or redistributed.
- RTL/LTR and mixed-direction rendering must be tested.
- Server-side authorization, parameterized queries, CSRF protection for cookie-auth mutations, secure cookies, security headers, rate limits, and audit events are mandatory.
- Translation providers may be local or remote; credentials never enter browser code or repository files.

---

### Task 1: Establish application contracts and module layout

**Files:**
- Create: `ArchaeologicalKnowledgeSystem/README.md`
- Create: `ArchaeologicalKnowledgeSystem/contracts/document.schema.json`
- Create: `ArchaeologicalKnowledgeSystem/contracts/language.schema.json`
- Create: `ArchaeologicalKnowledgeSystem/contracts/ocr.schema.json`
- Create: `ArchaeologicalKnowledgeSystem/contracts/transliteration.schema.json`
- Create: `ArchaeologicalKnowledgeSystem/contracts/translation.schema.json`
- Create: `ArchaeologicalKnowledgeSystem/contracts/reasoning.schema.json`
- Create: `ArchaeologicalKnowledgeSystem/contracts/pdf-report.schema.json`
- Create: `ArchaeologicalKnowledgeSystem/data/README.md`
- Test: `ArchaeologicalKnowledgeSystem/tests/contracts/test_json_schemas.py`

**Interfaces:**
- Produces JSON contracts used by every language implementation.
- `translation.schema.json` requires `source_text`, `source_language`, `scholarly_transliteration`, `english_transliteration`, `arabic_transliteration`, `target_language`, `translation`, `reasoning`, `confidence`, and `evidence_refs`.

- [ ] **Step 1: Write failing schema-validation fixtures.**
- [ ] **Step 2: Run `pytest ArchaeologicalKnowledgeSystem/tests/contracts -v` and verify failure.**
- [ ] **Step 3: Implement the schemas and minimal README/module map.**
- [ ] **Step 4: Run the contract tests and verify valid/invalid fixtures behave as specified.**
- [ ] **Step 5: Commit `feat: establish archaeological document and translation contracts`.**

### Task 2: Implement PDF reading and page model

**Files:**
- Create: `ArchaeologicalKnowledgeSystem/ingestion/pdf/reader.py`
- Create: `ArchaeologicalKnowledgeSystem/ingestion/pdf/models.py`
- Create: `ArchaeologicalKnowledgeSystem/ingestion/pdf/render.py`
- Create: `ArchaeologicalKnowledgeSystem/ingestion/pdf/security.py`
- Test: `ArchaeologicalKnowledgeSystem/tests/pdf/test_reader.py`
- Test: `ArchaeologicalKnowledgeSystem/tests/pdf/test_render.py`
- Test: `ArchaeologicalKnowledgeSystem/tests/pdf/test_security.py`

**Interfaces:**
- `read_pdf(path_or_bytes) -> DocumentRecord`
- `extract_page_text(document, page_number) -> PageText`
- `render_page(document, page_number, dpi=200) -> RenderedPage`
- `needs_ocr(page) -> bool`
- `validate_pdf_limits(bytes, limits) -> None`

- [ ] **Step 1: Create fixtures for text PDFs, image-only PDFs, malformed PDFs, and oversized page/object cases.**
- [ ] **Step 2: Write tests for page order, text extraction, image-only detection, bounded resource use, and provenance.**
- [ ] **Step 3: Implement an adapter boundary around the selected PDF reader/rendering engine rather than coupling the domain model to one vendor.**
- [ ] **Step 4: Run PDF tests and verify all fixtures pass.**
- [ ] **Step 5: Commit `feat: add secure archaeological PDF reader`.**

### Task 3: Integrate scanned-PDF OCR with the existing Thamudic scanner

**Files:**
- Create: `ArchaeologicalKnowledgeSystem/thamudic/scanner/pdf_adapter.py`
- Create: `ArchaeologicalKnowledgeSystem/thamudic/scanner/preprocess.py`
- Create: `ArchaeologicalKnowledgeSystem/thamudic/scanner/regions.py`
- Create: `ArchaeologicalKnowledgeSystem/thamudic/scanner/glyphs.py`
- Create: `ArchaeologicalKnowledgeSystem/thamudic/scanner/orientation.py`
- Modify: `data/ancient_north_arabian/alphabet.json` only if a verified metadata gap is discovered
- Test: `ArchaeologicalKnowledgeSystem/tests/thamudic/test_pdf_scan_pipeline.py`
- Test: `ArchaeologicalKnowledgeSystem/tests/thamudic/test_preprocessing.py`

**Interfaces:**
- `scan_pdf_inscription(page_image, script_id, options) -> OCRObservation`
- `preprocess_inscription(image, options) -> ImageArtifact`
- `detect_inscription_regions(image) -> list[Region]`
- `segment_glyphs(region) -> list[GlyphBox]`
- `generate_glyph_hypotheses(box, registry) -> list[GlyphHypothesis]`

- [ ] **Step 1: Build golden fixtures covering grayscale, CLAHE, adaptive thresholding, denoising, edge extraction, bounding boxes, RTL/LTR, and boustrophedon.**
- [ ] **Step 2: Write failing pipeline tests that require persisted page/region/glyph provenance.**
- [ ] **Step 3: Implement adapters using the existing scanner algorithms and canonical registry.**
- [ ] **Step 4: Add manual correction data structures without changing the original OCR observation.**
- [ ] **Step 5: Run the scanner test suite.**
- [ ] **Step 6: Commit `feat: connect PDF scans to ancient-script OCR pipeline`.**

### Task 4: Build source-language identification and Unicode/transliteration layer

**Files:**
- Create: `ArchaeologicalKnowledgeSystem/thamudic/languages/registry.py`
- Create: `ArchaeologicalKnowledgeSystem/thamudic/languages/capabilities.py`
- Create: `ArchaeologicalKnowledgeSystem/thamudic/text/unicode.py`
- Create: `ArchaeologicalKnowledgeSystem/thamudic/text/transliteration.py`
- Create: `ArchaeologicalKnowledgeSystem/thamudic/text/direction.py`
- Create: `ArchaeologicalKnowledgeSystem/data/languages/ancient.json`
- Create: `ArchaeologicalKnowledgeSystem/data/languages/modern-bcp47.json`
- Test: `ArchaeologicalKnowledgeSystem/tests/languages/test_registry.py`
- Test: `ArchaeologicalKnowledgeSystem/tests/languages/test_transliteration.py`
- Test: `ArchaeologicalKnowledgeSystem/tests/languages/test_bidi.py`

**Interfaces:**
- `LanguageRegistry.get(language_id) -> LanguageCapability`
- `normalize_source_text(text) -> NormalizedText`
- `transliterate(source_text, source_language, convention) -> TransliterationRecord`
- `to_english_readable(transliteration) -> str`
- `to_arabic_readable(transliteration) -> str`

- [ ] **Step 1: Create registry fixtures for Dadanitic, Safaitic, Hismaic, Taymanitic, Minaic, Thamudic variants and additional planned scripts.**
- [ ] **Step 2: Test NFC, code points, UTF-8 bytes, grapheme handling, RTL/LTR and mixed text.**
- [ ] **Step 3: Implement registry capability checks and transliteration adapters.**
- [ ] **Step 4: Ensure unsupported combinations return explicit capability errors rather than invented output.**
- [ ] **Step 5: Run language tests.**
- [ ] **Step 6: Commit `feat: add registry-driven ancient language and transliteration layer`.**

### Task 5: Implement semantic reading and translation reasoning

**Files:**
- Create: `ArchaeologicalKnowledgeSystem/intelligence/translation/pipeline.py`
- Create: `ArchaeologicalKnowledgeSystem/intelligence/translation/lexicon.py`
- Create: `ArchaeologicalKnowledgeSystem/intelligence/translation/morphology.py`
- Create: `ArchaeologicalKnowledgeSystem/intelligence/translation/context.py`
- Create: `ArchaeologicalKnowledgeSystem/intelligence/translation/reasoning.py`
- Create: `ArchaeologicalKnowledgeSystem/intelligence/translation/providers.py`
- Create: `ArchaeologicalKnowledgeSystem/intelligence/translation/provenance.py`
- Test: `ArchaeologicalKnowledgeSystem/tests/translation/test_pipeline.py`
- Test: `ArchaeologicalKnowledgeSystem/tests/translation/test_reasoning.py`
- Test: `ArchaeologicalKnowledgeSystem/tests/translation/test_uncertainty.py`

**Interfaces:**
- `translate_object_inscription(object_id, target_language, options) -> TranslationBundle`
- `reason_reading(source_text, transliteration, lexical_candidates, context) -> ReasoningRecord`
- `translate_text(normalized_text, source_language, target_language, context) -> TranslationRecord`
- `compare_readings(readings) -> ReadingComparison`

- [ ] **Step 1: Create deterministic fixtures where glyph alternatives lead to different transliterations and translations.**
- [ ] **Step 2: Write tests requiring English transliteration, Arabic transliteration, target translation, reasoning, confidence, and evidence references.**
- [ ] **Step 3: Implement lexical/morphological/contextual reasoning as evidence-ranked stages.**
- [ ] **Step 4: Add local/remote provider interfaces with server-side configuration and a reference-mode fallback for languages without reliable MT.**
- [ ] **Step 5: Verify the system preserves competing interpretations instead of silently collapsing them.**
- [ ] **Step 6: Run translation tests.**
- [ ] **Step 7: Commit `feat: add evidence-aware ancient language reasoning and translation`.**

### Task 6: Add archaeological object translation record and database persistence

**Files:**
- Create: `ArchaeologicalKnowledgeSystem/database/sql/sqlite/001_initial.sql`
- Create: `ArchaeologicalKnowledgeSystem/database/sql/postgresql/001_initial.sql`
- Create: `ArchaeologicalKnowledgeSystem/database/sql/mysql/001_initial.sql`
- Create: `ArchaeologicalKnowledgeSystem/database/sql/sqlserver/001_initial.sql`
- Create: `ArchaeologicalKnowledgeSystem/database/sql/oracle/001_initial.sql`
- Create: `ArchaeologicalKnowledgeSystem/database/sql/access/001_initial.sql`
- Create: `ArchaeologicalKnowledgeSystem/database/json/object.json`
- Create: `ArchaeologicalKnowledgeSystem/database/repository.py`
- Create: `ArchaeologicalKnowledgeSystem/database/migrations/README.md`
- Test: `ArchaeologicalKnowledgeSystem/tests/database/test_translation_persistence.py`

**Interfaces:**
- `save_ocr_observation(observation) -> id`
- `save_transliteration(record) -> id`
- `save_translation(bundle) -> id`
- `get_object_translation(object_id, target_language) -> TranslationBundle`

- [ ] **Step 1: Write migration tests for object, inscription, OCR, transliteration, translation, reasoning, evidence, and provenance records.**
- [ ] **Step 2: Implement normalized tables and foreign-key relationships.**
- [ ] **Step 3: Add portable JSON serialization.**
- [ ] **Step 4: Validate parameterized queries and transaction behavior.**
- [ ] **Step 5: Run database tests against SQLite and schema validation for other dialects.**
- [ ] **Step 6: Commit `feat: persist archaeological OCR and translation evidence`.**

### Task 7: Generate multilingual archaeological PDFs

**Files:**
- Create: `ArchaeologicalKnowledgeSystem/export/pdf/models.py`
- Create: `ArchaeologicalKnowledgeSystem/export/pdf/templates.py`
- Create: `ArchaeologicalKnowledgeSystem/export/pdf/fonts.py`
- Create: `ArchaeologicalKnowledgeSystem/export/pdf/writer.py`
- Create: `ArchaeologicalKnowledgeSystem/export/pdf/object_report.py`
- Create: `ArchaeologicalKnowledgeSystem/export/pdf/inscription_report.py`
- Test: `ArchaeologicalKnowledgeSystem/tests/pdf/test_generation.py`
- Test: `ArchaeologicalKnowledgeSystem/tests/pdf/test_rtl_generation.py`

**Interfaces:**
- `generate_object_report(object_record, translation_bundle, output_path) -> PdfReport`
- `generate_inscription_report(inscription_record, output_path) -> PdfReport`
- `generate_collection_report(query, target_language, output_path) -> PdfReport`

- [ ] **Step 1: Create expected-output fixtures for English, Arabic, and mixed RTL/LTR reports.**
- [ ] **Step 2: Test that generated PDFs contain original source Unicode, scholarly transliteration, English transliteration, Arabic transliteration, target translation, reasoning, confidence, and citations.**
- [ ] **Step 3: Implement report templates with embedded/available fonts and deterministic metadata.**
- [ ] **Step 4: Add image/IIIF metadata and provenance timeline sections without copying restricted remote images unless rights permit.**
- [ ] **Step 5: Validate generated PDFs by reopening and extracting their text.**
- [ ] **Step 6: Run PDF generation tests.**
- [ ] **Step 7: Commit `feat: generate multilingual archaeological research PDFs`.**

### Task 8: Add JavaScript/TypeScript/Node document APIs

**Files:**
- Create: `ArchaeologicalKnowledgeSystem/web/typescript/contracts.ts`
- Create: `ArchaeologicalKnowledgeSystem/web/nodejs/pdf/reader.ts`
- Create: `ArchaeologicalKnowledgeSystem/web/nodejs/pdf/reports.ts`
- Create: `ArchaeologicalKnowledgeSystem/web/nodejs/translation/objectTranslation.ts`
- Create: `ArchaeologicalKnowledgeSystem/web/workers/pdf.worker.ts`
- Create: `ArchaeologicalKnowledgeSystem/web/javascript/translation-view.js`
- Test: `ArchaeologicalKnowledgeSystem/tests/web/test_document_api_contracts.ts`

**Interfaces:**
- `readPdf(request): Promise<DocumentRecord>`
- `generateObjectPdf(request): Promise<Uint8Array>`
- `getObjectTranslation(request): Promise<TranslationBundle>`

- [ ] **Step 1: Write TypeScript contract tests against JSON fixtures.**
- [ ] **Step 2: Implement Node adapters that call the shared service contracts rather than duplicating OCR logic.**
- [ ] **Step 3: Move expensive page rendering/OCR work into a Worker boundary.**
- [ ] **Step 4: Add browser translation rendering for source, transliterations, target translation, and reasoning.**
- [ ] **Step 5: Run Node/TypeScript tests and type-check.**
- [ ] **Step 6: Commit `feat: expose PDF and translation services to web clients`.**

### Task 9: Add PHP API/document integration

**Files:**
- Create: `ArchaeologicalKnowledgeSystem/php/api/PdfController.php`
- Create: `ArchaeologicalKnowledgeSystem/php/api/TranslationController.php`
- Create: `ArchaeologicalKnowledgeSystem/php/services/PdfService.php`
- Create: `ArchaeologicalKnowledgeSystem/php/services/TranslationService.php`
- Create: `ArchaeologicalKnowledgeSystem/php/contracts/translation.php`
- Test: `ArchaeologicalKnowledgeSystem/tests/php/TranslationControllerTest.php`

**Interfaces:**
- `POST /api/pdf/read`
- `POST /api/pdf/report`
- `GET /api/objects/{id}/translation?target_language=<BCP47>`

- [ ] **Step 1: Write API contract tests including authorization and invalid target-language cases.**
- [ ] **Step 2: Implement controllers with strict validation and parameterized persistence.**
- [ ] **Step 3: Add CSRF/session/security-header handling for cookie-authenticated mutations.**
- [ ] **Step 4: Run PHP tests.**
- [ ] **Step 5: Commit `feat: add PHP archaeological document and translation API`.**

### Task 10: Build the object translation/scanning UI

**Files:**
- Create: `ArchaeologicalKnowledgeSystem/web/html/object-translation.html`
- Create: `ArchaeologicalKnowledgeSystem/web/html/pdf-workbench.html`
- Create: `ArchaeologicalKnowledgeSystem/web/javascript/pdf-workbench.js`
- Create: `ArchaeologicalKnowledgeSystem/web/javascript/object-translation.js`
- Create: `ArchaeologicalKnowledgeSystem/web/css/research.css`
- Test: `ArchaeologicalKnowledgeSystem/tests/web/translation-ui.contract.test.js`

**Interfaces:**
- Object translation screen displays source text, glyph hypotheses, scholarly transliteration, English transliteration, Arabic transliteration, target translation, reasoning, alternatives, confidence, and evidence.
- PDF workbench displays page thumbnails, OCR regions, editable glyph boxes, source text, and export controls.

- [ ] **Step 1: Create UI fixtures and assertions for all required translation fields.**
- [ ] **Step 2: Implement object translation view with target-language selector driven by BCP-47 registry.**
- [ ] **Step 3: Implement PDF upload/read/scan workflow with worker-based processing.**
- [ ] **Step 4: Add manual OCR correction and save-as-new-observation behavior.**
- [ ] **Step 5: Add PDF report export and provenance display.**
- [ ] **Step 6: Run UI contract tests and browser smoke tests.**
- [ ] **Step 7: Commit `feat: add archaeological PDF and translation workbench`.**

### Task 11: Documentation, examples, and deployment

**Files:**
- Modify: `ArchaeologicalKnowledgeSystem/README.md`
- Create: `ArchaeologicalKnowledgeSystem/docs/PDF_READING.md`
- Create: `ArchaeologicalKnowledgeSystem/docs/PDF_GENERATION.md`
- Create: `ArchaeologicalKnowledgeSystem/docs/SOURCE_LANGUAGE_REASONING.md`
- Create: `ArchaeologicalKnowledgeSystem/docs/TRANSLATION_OUTPUT.md`
- Create: `ArchaeologicalKnowledgeSystem/docs/ANCIENT_LANGUAGE_CAPABILITIES.md`
- Create: `ArchaeologicalKnowledgeSystem/docs/API.md`
- Create: `ArchaeologicalKnowledgeSystem/docs/SECURITY.md`
- Create: `ArchaeologicalKnowledgeSystem/examples/object-translation.json`
- Create: `ArchaeologicalKnowledgeSystem/examples/pdf-report.json`

- [ ] **Step 1: Document PDF security/resource limits, supported reader modes, OCR pipeline, and report generation.**
- [ ] **Step 2: Document the exact translation output contract and meaning-reasoning evidence model.**
- [ ] **Step 3: Document capability levels for ancient languages and explicitly distinguish reference reasoning from reliable MT.**
- [ ] **Step 4: Document local and remote provider configuration without embedding secrets.**
- [ ] **Step 5: Add end-to-end examples and deployment commands.**
- [ ] **Step 6: Commit `docs: document PDF, transliteration, translation and reasoning workflows`.**

### Task 12: End-to-end verification and integration

**Files:**
- Create: `ArchaeologicalKnowledgeSystem/tests/e2e/test_pdf_to_translation.py`
- Create: `ArchaeologicalKnowledgeSystem/tests/fixtures/README.md`
- Modify: `ArchaeologicalKnowledgeSystem/README.md` if verification reveals missing setup instructions

**Interfaces:**
- End-to-end path: PDF/image -> page model -> source-language detection -> OCR -> glyph alternatives -> Unicode -> scholarly transliteration -> English transliteration -> Arabic transliteration -> target translation -> reasoning/evidence -> object persistence -> multilingual PDF report.

- [ ] **Step 1: Build deterministic synthetic fixtures for supported ancient-script workflows and mixed-direction text.**
- [ ] **Step 2: Run the full Python test suite and contract validation.**
- [ ] **Step 3: Run Node/TypeScript tests and type checking.**
- [ ] **Step 4: Run PHP tests/static validation.**
- [ ] **Step 5: Reopen generated PDFs and verify extracted text contains every required translation layer.**
- [ ] **Step 6: Verify no credentials, generated secrets, or restricted source images were committed.**
- [ ] **Step 7: Verify the final repository tree and documentation links.**
- [ ] **Step 8: Commit `test: verify end-to-end archaeological PDF translation workflow`.**

## Coverage checklist

- PDF reading: Tasks 2-3
- Scanned PDF OCR: Task 3
- Thamudic scanner features: Task 3
- Ancient-language registry: Task 4
- Unicode/UTF-8: Task 4
- English transliteration: Tasks 4-5
- Arabic transliteration: Tasks 4-5
- Target-language translation: Task 5
- Meaning/reasoning: Task 5
- Provenance/evidence: Tasks 5-6
- Object persistence: Task 6
- PDF generation: Task 7
- JavaScript/TypeScript/Node: Task 8
- PHP: Task 9
- UI: Task 10
- Documentation: Task 11
- End-to-end verification: Task 12

## External design references

The implementation plan allows interchangeable PDF adapters because PDF extraction and scanned-document handling vary by engine. Current ecosystem research confirms that PDF extraction must distinguish embedded text from image-only pages, and that ancient-language OCR/transliteration benefits from explicit evidence-aware pipelines rather than assuming generic OCR is sufficient. citeturn0search9turn0search7turn0search11
