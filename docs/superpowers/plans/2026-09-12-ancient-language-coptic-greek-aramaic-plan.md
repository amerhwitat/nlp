# Ancient Language Intelligence Expansion — Coptic Greek Aramaic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend ArchaeologicalKnowledgeSystem so Coptic, the full historical Greek continuum, and Ancient-to-Modern Aramaic can be registered, recognized, encoded, transliterated, reasoned over, persisted, and documented alongside the existing ancient-language system.

**Architecture:** Keep the registry-driven architecture and separate language, historical stage, dialect, script, orthography, Unicode representation, and model capability. Add reusable Unicode ingestion, language-stage classification, mixed-script span detection, script-specific adapters, provenance-aware ER/MADM persistence, and versioned transliteration/model interfaces rather than hard-coded language branches.

**Tech Stack:** Python 3.x registry/data tooling; JSON/JSON Schema; existing C++/Python/.NET/VC++ components in `nlp`; SQL-compatible OLTP schema; MADM/star-schema SQL; existing OCR/AI stack; Unicode Character Database and UAX #57 data; pytest plus existing repository test/build systems.

**Spec:** `docs/superpowers/specs/2026-09-12-ancient-language-intelligence-expansion.md` and `docs/superpowers/specs/2026-09-12-ancient-language-coptic-greek-aramaic-addendum.md`

## Global Constraints

- Unicode is the normative character-encoding layer; Unicode script identity must never be treated as language identity.
- Coptic must remain a distinct script identity even where characters historically overlap with Greek.
- Greek language stage, dialect, orthography, and script must be separate registry dimensions.
- Aramaic language stages and modern descendants must be distinct from Syriac/Mandaic/script identity while preserving their relationships.
- Every imported registry/model fact must retain source, version, provenance, and validation status.
- OCR and neural predictions must remain hypotheses with model/version/confidence metadata and must never silently become authoritative historical facts.
- Original source text must be preserved separately from normalized Unicode, transliteration, and translation outputs.
- RTL, bidi, shaping, combining marks, polytonic Greek, and mixed-script spans must survive round-trip tests.
- Unicode 18.0 data must be version-pinned in generated registry snapshots; draft/beta data must not be silently represented as immutable historical facts.
- No third-party image or corpus rights are implied by imported metadata.

---

## File and module map

**Create:**
- `data/ancient_languages/languages.json` — canonical language/stage/dialect registry.
- `data/ancient_languages/scripts.json` — script, script-variant, direction, shaping, and ISO 15924 metadata.
- `data/ancient_languages/orthographies.json` — Greek polytonic/monotonic and other orthographic profiles.
- `data/ancient_languages/unicode.json` — generated Unicode character/code-point/UTF-8 metadata index.
- `data/ancient_languages/transliteration.json` — versioned scholarly transliteration profiles.
- `data/ancient_languages/capabilities.json` — OCR/segmentation/language-ID/morphology/translation capability declarations.
- `data/ancient_languages/sources.json` — source authority/version/license/provenance records.
- `schemas/ancient_languages.schema.json` — JSON Schema for registry records.
- `python/ancient_languages/registry.py` — typed registry loader/validator.
- `python/ancient_languages/unicode_registry.py` — Unicode/UCD ingestion and UTF-8 derivation.
- `python/ancient_languages/stage_classifier.py` — language-stage/dialect classification interface.
- `python/ancient_languages/mixed_script.py` — span-level mixed-script detector.
- `python/ancient_languages/transliteration.py` — transliteration profile execution.
- `python/ancient_languages/provenance.py` — evidence/source/model provenance types.
- `python/ancient_languages/recognition.py` — hierarchical recognition orchestration.
- `python/tests/test_ancient_language_registry.py` — registry validation tests.
- `python/tests/test_unicode_registry.py` — Unicode/UTF-8 tests.
- `python/tests/test_greek_coptic_aramaic.py` — language/script/stage tests.
- `python/tests/test_mixed_script.py` — mixed-span recognition tests.
- `python/tests/test_transliteration.py` — transliteration provenance/round-trip tests.
- `db/sql/ancient_language_schema.sql` — normalized ER tables and constraints.
- `db/sql/ancient_language_madm.sql` — MADM dimensions/facts.
- `db/sql/ancient_language_seeds.sql` — deterministic registry seed references.
- `docs/ANCIENT_LANGUAGES.md` — user/developer language coverage guide.
- `docs/UNICODE_ANCIENT_SCRIPTS.md` — Unicode ingestion and UTF-8 guide.
- `docs/ANCIENT_LANGUAGE_AI.md` — recognition/model architecture.
- `docs/ANCIENT_LANGUAGE_ER_MADM.md` — ER and analytical model.
- `docs/TRANSLITERATION_TRANSLATION.md` — transliteration/translation behavior.
- `tests/fixtures/ancient_languages/` — representative Greek, Coptic, Aramaic, Syriac and Mandaic fixtures.

**Modify:**
- `README.md` — feature and supported-language overview.
- `docs/superpowers/specs/2026-09-12-ancient-language-intelligence-expansion.md` — incorporate the approved addendum into the canonical specification.
- Existing OCR/Thamudic registry documentation and test entry points — connect them through adapters instead of duplicate registries.
- Existing build/test workflows — validate registry generation and Python/database tests in CI.

---

### Task 1: Consolidate the approved Coptic/Greek/Aramaic specification

**Files:**
- Modify: `docs/superpowers/specs/2026-09-12-ancient-language-intelligence-expansion.md`
- Reference: `docs/superpowers/specs/2026-09-12-ancient-language-coptic-greek-aramaic-addendum.md`
- Test: `docs/superpowers/plans/2026-09-12-ancient-language-coptic-greek-aramaic-plan.md` self-review checklist

**Interfaces:**
- Produces the canonical requirements consumed by every later task.

- [ ] **Step 1: Merge the addendum sections into the canonical specification.**

Include explicit requirements for Coptic, Greek historical stages/dialects/orthographies, Aramaic ancient-to-modern stages, Syriac/Mandaic script relationships, mixed-script spans, Unicode 18.0 versioning, new ER/MADM dimensions, and validation requirements.

- [ ] **Step 2: Verify the canonical specification contains no contradictory language/script assumptions.**

Run a text search for `Coptic`, `Greek`, `Aramaic`, `Mandaic`, `Syriac`, `LANGUAGE_STAGE`, `DIALECT`, `ORTHOGRAPHY`, and `MODERN_CONTINUATION`; each must occur in the appropriate scope/ER/validation sections.

- [ ] **Step 3: Commit.**

```bash
git add docs/superpowers/specs/2026-09-12-ancient-language-intelligence-expansion.md docs/superpowers/specs/2026-09-12-ancient-language-coptic-greek-aramaic-addendum.md
git commit -m "docs: consolidate ancient language expansion requirements"
```

---

### Task 2: Build the registry schema and historical language model

**Files:**
- Create: `schemas/ancient_languages.schema.json`
- Create: `data/ancient_languages/languages.json`
- Create: `data/ancient_languages/scripts.json`
- Create: `data/ancient_languages/orthographies.json`
- Create: `data/ancient_languages/capabilities.json`
- Create: `data/ancient_languages/sources.json`
- Create: `python/ancient_languages/registry.py`
- Test: `python/tests/test_ancient_language_registry.py`

**Interfaces:**
- `load_registry(path: str) -> AncientLanguageRegistry`
- `AncientLanguageRegistry.get_language(language_id: str) -> LanguageRecord`
- `AncientLanguageRegistry.get_stage(stage_id: str) -> LanguageStage`
- `AncientLanguageRegistry.get_script(script_id: str) -> ScriptRecord`
- `AncientLanguageRegistry.validate() -> list[RegistryIssue]`

- [ ] **Step 1: Write failing tests for language-stage-script relationships.**

```python
def test_coptic_is_distinct_from_greek():
    registry = load_registry("data/ancient_languages")
    assert registry.get_script("coptic").unicode_script == "Copt"
    assert registry.get_script("grek").unicode_script == "Grek"
    assert registry.get_script("coptic").id != registry.get_script("grek").id
```

```python
def test_aramaic_historical_and_modern_stages_are_related_but_distinct():
    registry = load_registry("data/ancient_languages")
    ancient = registry.get_stage("aramaic.imperial")
    modern = registry.get_stage("neo_aramaic")
    assert ancient.language_id == "aramaic"
    assert ancient.id != modern.id
    assert "modern_continuation" in modern.relationships
```

- [ ] **Step 2: Run the targeted test and verify it fails because the registry is absent.**

Run: `pytest python/tests/test_ancient_language_registry.py -v`

Expected: FAIL because the new registry modules/data do not yet exist.

- [ ] **Step 3: Implement the schema and seed records.**

Seed Coptic, Greek historical stages, Aramaic historical stages, Syriac, Mandaic, and modern continuation relationships. Include existing Mesopotamian/Egyptian/Arabian registry families through references or migrated records so the new registry remains one canonical source.

- [ ] **Step 4: Run the tests.**

Run: `pytest python/tests/test_ancient_language_registry.py -v`

Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add schemas/ancient_languages.schema.json data/ancient_languages python/ancient_languages/registry.py python/tests/test_ancient_language_registry.py
git commit -m "feat: add historical ancient language registry"
```

---

### Task 3: Implement Unicode and UTF-8 registry ingestion

**Files:**
- Create: `data/ancient_languages/unicode.json`
- Create: `python/ancient_languages/unicode_registry.py`
- Test: `python/tests/test_unicode_registry.py`

**Interfaces:**
- `codepoint_to_utf8(codepoint: int) -> bytes`
- `utf8_to_codepoints(value: bytes) -> list[int]`
- `load_unicode_registry(path: str, version: str) -> UnicodeRegistry`
- `UnicodeRegistry.lookup(codepoint: int) -> UnicodeCharacter`

- [ ] **Step 1: Write failing UTF-8 and normalization tests.**

```python
def test_utf8_round_trip():
    for text in ["Ἀρχή", "ⲁⲛⲟⲕ", "𐡀"]:
        encoded = text.encode("utf-8")
        assert encoded.decode("utf-8") == text
```

```python
def test_registry_preserves_unicode_script_identity():
    registry = load_unicode_registry("data/ancient_languages/unicode.json", "18.0.0")
    assert registry.script_for("ⲁ") == "Copt"
```

- [ ] **Step 2: Run tests and verify failure.**

Run: `pytest python/tests/test_unicode_registry.py -v`

Expected: FAIL until the registry implementation exists.

- [ ] **Step 3: Implement deterministic UCD import and UTF-8 derivation.**

Store code point, Unicode name, block, Script, General_Category, Bidi_Class, combining properties, normalization data, UTF-8 bytes, Unicode release, and source checksum. Do not infer language from Script alone.

- [ ] **Step 4: Run tests.**

Run: `pytest python/tests/test_unicode_registry.py -v`

Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add data/ancient_languages/unicode.json python/ancient_languages/unicode_registry.py python/tests/test_unicode_registry.py
git commit -m "feat: add versioned Unicode ancient-script registry"
```

---

### Task 4: Add Greek historical stages and orthography profiles

**Files:**
- Modify: `data/ancient_languages/languages.json`
- Modify: `data/ancient_languages/orthographies.json`
- Modify: `data/ancient_languages/transliteration.json`
- Create: `python/tests/test_greek_coptic_aramaic.py`

**Interfaces:**
- `registry.get_stage("greek.archaic")`
- `registry.get_stage("greek.classical")`
- `registry.get_stage("greek.koine")`
- `registry.get_stage("greek.byzantine")`
- `registry.get_stage("greek.modern")`
- `transliterate(text: str, profile_id: str) -> TransliterationResult`

- [ ] **Step 1: Write failing Greek stage/orthography tests.**

```python
def test_greek_historical_stages_are_distinct():
    registry = load_registry("data/ancient_languages")
    ids = ["greek.archaic", "greek.classical", "greek.koine", "greek.byzantine", "greek.modern"]
    assert len({registry.get_stage(x).id for x in ids}) == len(ids)
```

```python
def test_polytonic_profile_is_not_language_identity():
    registry = load_registry("data/ancient_languages")
    assert registry.get_orthography("greek.polytonic").language_stage_ids
    assert registry.get_orthography("greek.polytonic").script_id == "grek"
```

- [ ] **Step 2: Run the targeted tests and verify failure.**

Run: `pytest python/tests/test_greek_coptic_aramaic.py -k greek -v`

Expected: FAIL before the Greek records are added.

- [ ] **Step 3: Add historical Greek stages, major dialect metadata, Linear B linkage where applicable, polytonic/monotonic orthographies, historic letters, numeral metadata, and transliteration profiles.**

- [ ] **Step 4: Run tests and verify PASS.**

Run: `pytest python/tests/test_greek_coptic_aramaic.py -k greek -v`

- [ ] **Step 5: Commit.**

```bash
git add data/ancient_languages python/tests/test_greek_coptic_aramaic.py
git commit -m "feat: model Greek historical stages and orthographies"
```

---

### Task 5: Add Coptic and Aramaic/Syriac/Mandaic lineage models

**Files:**
- Modify: `data/ancient_languages/languages.json`
- Modify: `data/ancient_languages/scripts.json`
- Modify: `data/ancient_languages/orthographies.json`
- Modify: `data/ancient_languages/capabilities.json`
- Test: `python/tests/test_greek_coptic_aramaic.py`

**Interfaces:**
- `registry.get_stage("coptic.sahidic")`
- `registry.get_stage("coptic.bohairic")`
- `registry.get_stage("aramaic.ancient")`
- `registry.get_stage("aramaic.imperial")`
- `registry.get_stage("aramaic.biblical")`
- `registry.get_stage("syriac.classical")`
- `registry.get_stage("mandaic.classical")`
- `registry.get_stage("neo_aramaic")`

- [ ] **Step 1: Write failing lineage and RTL tests.**

```python
def test_coptic_varieties_are_language_stages():
    registry = load_registry("data/ancient_languages")
    assert registry.get_stage("coptic.sahidic").language_id == "coptic"
    assert registry.get_stage("coptic.bohairic").language_id == "coptic"
```

```python
def test_aramaic_related_scripts_keep_distinct_script_ids():
    registry = load_registry("data/ancient_languages")
    assert registry.get_script("imperial_aramaic").direction == "rtl"
    assert registry.get_script("syriac").direction == "rtl"
    assert registry.get_script("mandaic").direction == "rtl"
```

- [ ] **Step 2: Run tests and verify failure.**

Run: `pytest python/tests/test_greek_coptic_aramaic.py -k 'coptic or aramaic' -v`

Expected: FAIL before lineage records exist.

- [ ] **Step 3: Implement Coptic varieties and Aramaic historical/modern branches.**

Include Imperial/Ancient/Biblical/Official Aramaic, Nabataean/Palmyrene/Hatran relationships, Syriac and Mandaic script traditions, Samaritan Aramaic, and modern Neo-Aramaic continuation links. Store bidi/shaping/vocalization capability metadata.

- [ ] **Step 4: Run tests.**

Run: `pytest python/tests/test_greek_coptic_aramaic.py -k 'coptic or aramaic' -v`

Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add data/ancient_languages python/tests/test_greek_coptic_aramaic.py
git commit -m "feat: add Coptic and Aramaic language lineages"
```

---

### Task 6: Implement transliteration and provenance-aware language analysis

**Files:**
- Create: `data/ancient_languages/transliteration.json`
- Create: `python/ancient_languages/transliteration.py`
- Create: `python/ancient_languages/stage_classifier.py`
- Create: `python/ancient_languages/provenance.py`
- Test: `python/tests/test_transliteration.py`

**Interfaces:**
- `transliterate(text: str, profile_id: str) -> TransliterationResult`
- `classify_stage(observation: LanguageObservation) -> list[StageHypothesis]`
- `make_provenance(source_id: str, evidence_class: str, confidence: float) -> ProvenanceRecord`

- [ ] **Step 1: Write failing transliteration/provenance tests.**

```python
def test_transliteration_records_profile_and_provenance():
    result = transliterate("Ἀρχή", "greek.scholarly")
    assert result.profile_id == "greek.scholarly"
    assert result.provenance.evidence_class == "scholarly_assertion"
```

```python
def test_uncertain_stage_returns_ranked_hypotheses():
    hypotheses = classify_stage(LanguageObservation(text="sample", script="Grek"))
    assert all(0.0 <= h.confidence <= 1.0 for h in hypotheses)
```

- [ ] **Step 2: Run tests and verify failure.**

Run: `pytest python/tests/test_transliteration.py -v`

Expected: FAIL until the interfaces exist.

- [ ] **Step 3: Implement deterministic profiles and provenance records.**

Profiles must identify source language/stage, target transliteration convention, version, rule set, source, and confidence. Never overwrite the original source text.

- [ ] **Step 4: Run tests and verify PASS.**

Run: `pytest python/tests/test_transliteration.py -v`

- [ ] **Step 5: Commit.**

```bash
git add data/ancient_languages/transliteration.json python/ancient_languages/stage_classifier.py python/ancient_languages/transliteration.py python/ancient_languages/provenance.py python/tests/test_transliteration.py
git commit -m "feat: add transliteration and provenance engine"
```

---

### Task 7: Add mixed-script and hierarchical recognition orchestration

**Files:**
- Create: `python/ancient_languages/mixed_script.py`
- Create: `python/ancient_languages/recognition.py`
- Test: `python/tests/test_mixed_script.py`

**Interfaces:**
- `detect_spans(text_or_glyphs: object) -> list[ScriptLanguageSpan]`
- `recognize_document(asset: RecognitionAsset) -> RecognitionResult`

- [ ] **Step 1: Write failing mixed-script tests.**

```python
def test_mixed_greek_coptic_aramaic_spans_are_independent():
    spans = detect_spans("Ἀρχή ⲁⲛⲟⲕ 𐡀𐡌𐡀")
    assert {s.script_id for s in spans} >= {"grek", "Copt", "imperial_aramaic"}
```

- [ ] **Step 2: Run the test and verify failure.**

Run: `pytest python/tests/test_mixed_script.py -v`

Expected: FAIL until span detection exists.

- [ ] **Step 3: Implement span-level detection and recognition pipeline orchestration.**

The recognizer must preserve candidate hypotheses for script, language stage, dialect, character/sign, Unicode, transliteration and translation. It must accept script-specific adapters without hard-coded language branches.

- [ ] **Step 4: Run tests.**

Run: `pytest python/tests/test_mixed_script.py -v`

Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add python/ancient_languages/mixed_script.py python/ancient_languages/recognition.py python/tests/test_mixed_script.py
git commit -m "feat: add mixed-script ancient language recognition"
```

---

### Task 8: Implement normalized ER schema and MADM analytical layer

**Files:**
- Create: `db/sql/ancient_language_schema.sql`
- Create: `db/sql/ancient_language_madm.sql`
- Create: `db/sql/ancient_language_seeds.sql`
- Test: `python/tests/test_ancient_language_registry.py`

**Interfaces:**
- Tables: `language`, `language_stage`, `dialect`, `script`, `script_variant`, `orthography`, `modern_continuation`, `language_attestation`, `unicode_character`, `transliteration_system`, `transliteration_rule`, `evidence_source`, `model_version`.
- MADM dimensions: `dim_period`, `dim_geography`, `dim_language`, `dim_language_stage`, `dim_dialect`, `dim_script`, `dim_orthography`, `dim_unicode`, `dim_source`, `dim_model`, `dim_confidence`.
- MADM facts: `fact_stage_classification`, `fact_glyph_recognition`, `fact_transliteration`, `fact_translation`, `fact_source_assertion`.

- [ ] **Step 1: Write schema-integrity tests against a temporary SQLite database.**

```python
def test_ancient_language_schema_creates_relationships(tmp_path):
    # Execute db/sql/ancient_language_schema.sql against SQLite and inspect required tables/FKs.
    assert required_tables_exist(tmp_path)
```

- [ ] **Step 2: Run and verify failure before schema creation.**

Run: `pytest python/tests/test_ancient_language_registry.py -k schema -v`

Expected: FAIL because the SQL schema is not present.

- [ ] **Step 3: Implement normalized tables, constraints, provenance fields, and MADM dimensions/facts.**

Use surrogate primary keys plus stable registry identifiers; enforce unique `(unicode_version, codepoint)` and unique stage/script identifiers; add validity-period, evidence-class, confidence, source, and model foreign keys.

- [ ] **Step 4: Run the schema tests.**

Run: `pytest python/tests/test_ancient_language_registry.py -k schema -v`

Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add db/sql python/tests/test_ancient_language_registry.py
git commit -m "feat: add ancient language ER and MADM schemas"
```

---

### Task 9: Connect existing Thamudic/Ancient North Arabian and Egyptian registries

**Files:**
- Modify: `data/ancient_languages/languages.json`
- Modify: `data/ancient_languages/scripts.json`
- Modify: existing `data/ancient_north_arabian/alphabet.json`
- Modify: existing Thamudic/OCR registry adapters under `python/thamudic/`
- Test: existing `python/tests/` plus new registry integration tests

**Interfaces:**
- `import_existing_script_registry(source_path: str) -> RegistryFragment`
- `merge_registry_fragments(*fragments: RegistryFragment) -> AncientLanguageRegistry`

- [ ] **Step 1: Add a failing integration test that loads the existing North Arabian registry through the new interface.**

```python
def test_north_arabian_registry_is_consumed_without_duplication():
    registry = load_registry("data/ancient_languages")
    assert registry.get_script("old_north_arabian")
    assert registry.external_sources["ancient_north_arabian/alphabet.json"]
```

- [ ] **Step 2: Run and verify failure.**

Run: `pytest python/tests -k north_arabian -v`

Expected: FAIL until the adapter exists.

- [ ] **Step 3: Implement adapter-based ingestion.**

Do not duplicate the canonical glyph table; retain the existing registry as a source fragment and expose it through the unified language/script interfaces. Connect Egyptian and other existing ancient registries similarly.

- [ ] **Step 4: Run the complete relevant Python suite.**

Run: `pytest python/tests -v`

Expected: PASS for existing and new ancient-language tests.

- [ ] **Step 5: Commit.**

```bash
git add data/ancient_languages data/ancient_north_arabian python/thamudic python/tests
git commit -m "refactor: connect existing ancient script registries"
```

---

### Task 10: Update documentation, APIs, CI, and end-to-end validation

**Files:**
- Modify: `README.md`
- Create: `docs/ANCIENT_LANGUAGES.md`
- Create: `docs/UNICODE_ANCIENT_SCRIPTS.md`
- Create: `docs/ANCIENT_LANGUAGE_AI.md`
- Create: `docs/ANCIENT_LANGUAGE_ER_MADM.md`
- Create: `docs/TRANSLITERATION_TRANSLATION.md`
- Modify: existing CI/test workflow files
- Test: full repository test/build entry points

**Interfaces:**
- Documentation must expose supported language stages, scripts, capabilities, evidence classes, registry generation, and database model.
- CI must run registry validation, Python tests, SQL schema tests, and existing repository tests.

- [ ] **Step 1: Add documentation checks for required language names and architecture terms.**

```python
def test_docs_cover_coptic_greek_aramaic():
    text = open("docs/ANCIENT_LANGUAGES.md", encoding="utf-8").read()
    for term in ["Coptic", "Greek", "Aramaic", "Syriac", "Mandaic", "polytonic"]:
        assert term in text
```

- [ ] **Step 2: Run documentation tests and verify failure before docs exist.**

Run: `pytest python/tests -k docs -v`

Expected: FAIL until documentation is created and wired into the test suite.

- [ ] **Step 3: Write complete documentation and CI integration.**

Document registry generation, Unicode version pinning, UTF-8 behavior, historical stages, mixed-script recognition, ER/MADM model, model capability levels, provenance, rights, transliteration and translation output. Add CI commands that execute registry/schema validation and the existing test/build matrix.

- [ ] **Step 4: Run all validation layers.**

Run:
```bash
pytest python/tests -v
```

Then execute the repository's existing C++/.NET/VC++ build and test commands without removing existing checks. Validate SQL creation for SQLite and the project's supported SQL targets where CI infrastructure exists.

Expected: all newly added tests pass and existing tests/builds remain green.

- [ ] **Step 5: Commit.**

```bash
git add README.md docs python/tests .github

git commit -m "docs: complete ancient language intelligence integration"
```

---

## Final verification gate

- [ ] Validate JSON schemas and all registry files.
- [ ] Validate Unicode/UTF-8 round trips for Greek, Coptic, Aramaic, Syriac, Mandaic, Egyptian and existing Ancient North Arabian records.
- [ ] Validate historical-stage relationships and modern-continuation relationships.
- [ ] Validate mixed-script span recognition.
- [ ] Validate transliteration provenance and competing hypotheses.
- [ ] Validate ER foreign keys and uniqueness constraints.
- [ ] Validate MADM dimension/fact foreign keys.
- [ ] Validate existing Thamudic/OCR behavior remains compatible.
- [ ] Run full repository test/build matrix.
- [ ] Review generated registry provenance and Unicode release metadata.
- [ ] Confirm documentation matches actual capability declarations and does not claim OCR/translation capability where only encoding/reference data exists.
