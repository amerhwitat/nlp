# Hebrew, Alphabet/UTF-8 and Voice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Ancient/Modern Hebrew, a reproducible alphabet/sign and Unicode/UTF-8 registry, and pronunciation/speech controls to the approved ancient-language architecture.

**Architecture:** Keep language, stage, dialect, script, character/sign, Unicode and pronunciation as separate registry dimensions. Ship a generated offline registry snapshot backed by deterministic UCD ingestion, plus a browser speech adapter and provenance-aware pronunciation profiles.

**Tech Stack:** Python 3.x, JSON/JSON Schema, pytest, Unicode UCD/code charts, ISO 15924 metadata, browser Web Speech API, existing JavaScript/TypeScript/PHP web layers.

**Spec:** `docs/superpowers/specs/2026-09-12-hebrew-alphabet-utf8-voice-addendum.md` plus `docs/superpowers/specs/2026-09-12-ancient-language-intelligence-expansion.md`.

## Global Constraints

- Unicode is normative for encoding; script is never treated as language identity.
- Registry facts retain source, version, checksum/provenance and validation state.
- Hebrew RTL and combining marks must round-trip.
- Ancient pronunciation must be labeled reconstructed/reference when historically uncertain.
- Speech controls must degrade gracefully when Web Speech voices are unavailable.
- Offline registry lookup must not require network access.

---

### Task 1: Hebrew registry and alphabet inventory

**Files:**
- Create: `data/ancient_languages/hebrew.json`
- Create: `data/ancient_languages/alphabet_tables.json`
- Create: `python/ancient_languages/hebrew.py`
- Create: `python/tests/test_hebrew.py`

**Interfaces:**
- `load_hebrew_registry(path: str) -> HebrewRegistry`
- `HebrewRegistry.stages() -> list[str]`
- `HebrewRegistry.alphabet(stage_id: str) -> list[dict]`
- `HebrewRegistry.lookup(char: str) -> dict | None`

- [ ] Write tests for Biblical/Ancient, Masoretic and Modern Hebrew stages, RTL direction, 22 core letters, final forms and combining marks.
- [ ] Run `pytest python/tests/test_hebrew.py -v` and confirm the new tests fail before implementation.
- [ ] Implement registry records with Paleo-Hebrew and Samaritan relationships explicitly separated from Hebrew block identity.
- [ ] Add UTF-8 bytes for every shipped Hebrew character using deterministic derivation and verify stored values.
- [ ] Run the targeted tests and commit `feat: add Hebrew historical registry and alphabet inventory`.

### Task 2: Unicode/UTF-8 generator and offline snapshot

**Files:**
- Create: `python/ancient_languages/unicode_registry.py`
- Create: `data/ancient_languages/unicode_snapshot.json`
- Create: `python/tests/test_unicode_snapshot.py`
- Create: `tools/generate_ancient_unicode_registry.py`

**Interfaces:**
- `codepoint_to_utf8(codepoint: int) -> bytes`
- `utf8_to_codepoints(value: bytes) -> list[int]`
- `generate_snapshot(ucd_root: str, output: str, version: str) -> None`
- `load_unicode_registry(path: str, version: str) -> UnicodeRegistry`

- [ ] Write failing round-trip tests for Greek, Coptic, Hebrew, Aramaic, Egyptian and cuneiform representatives.
- [ ] Run the targeted tests and confirm failure.
- [ ] Implement deterministic UCD parsing, UTF-8 derivation, normalization metadata, Script/Bidi/General_Category fields, release/status and source checksum.
- [ ] Add a reproducible offline snapshot manifest and ensure draft/beta Unicode releases are explicitly labeled.
- [ ] Run tests and commit `feat: add deterministic ancient Unicode UTF-8 registry`.

### Task 3: Pronunciation model and voice adapter

**Files:**
- Create: `data/ancient_languages/pronunciation.json`
- Create: `python/ancient_languages/pronunciation.py`
- Create: `python/tests/test_pronunciation.py`
- Create: `web/javascript/voice.js`

**Interfaces:**
- `resolve_pronunciation(text: str, language_stage: str, profile_id: str) -> PronunciationResult`
- `list_voice_capabilities() -> VoiceCapabilities`
- `SpeechController.play(text: str, options: SpeechOptions) -> bool`
- `SpeechController.pause() -> None`
- `SpeechController.resume() -> None`
- `SpeechController.stop() -> None`

- [ ] Write tests for native/modern vs reconstructed/reference pronunciation labels and missing-voice fallback.
- [ ] Run tests and confirm failure.
- [ ] Implement pronunciation profiles with IPA, grapheme-to-phoneme mappings, syllable/stress metadata, locale, backend, confidence and provenance.
- [ ] Implement Web Speech feature detection and safe speech-synthesis controls for play/pause/resume/stop/rate/pitch/volume/voice/locale/replay.
- [ ] Run tests and commit `feat: add provenance-aware pronunciation and speech adapter`.

### Task 4: Transliteration and translation Listen UI

**Files:**
- Modify: `web/javascript/index.html`
- Modify: `web/javascript/app.js`
- Create: `web/javascript/voice-controls.js`
- Create: `web/javascript/voice-controls.css`

**Interfaces:**
- `renderListenControl(target, controller, options) -> HTMLElement`
- `bindVoiceControls(container, controller) -> void`

- [ ] Add transliteration and translation text areas plus Listen controls directly below them.
- [ ] Add voice selector, locale, rate, pitch, volume, playback mode and pronunciation mode controls.
- [ ] Bind controls to `SpeechController` and preserve RTL/LTR direction from language metadata.
- [ ] Add graceful disabled/fallback state when speech synthesis is unavailable.
- [ ] Run browser/static validation and commit `feat: add transliteration and translation speech controls`.

### Task 5: ER/MADM and provenance schema

**Files:**
- Create: `db/sql/ancient_language_voice_schema.sql`
- Create: `db/sql/ancient_language_voice_madm.sql`
- Create: `docs/ANCIENT_LANGUAGE_VOICE.md`

- [ ] Add `ALPHABET`, `ALPHABET_MEMBER`, `UTF8_ENCODING`, `PRONUNCIATION_PROFILE`, `PRONUNCIATION_VARIANT`, `VOICE_PROFILE`, and `AUDIO_ASSET` with provenance constraints.
- [ ] Add analytical dimensions/facts for alphabet membership, Unicode/UTF-8, pronunciation, speech backend and confidence.
- [ ] Document reconstructed-vs-modern pronunciation semantics and UI behavior.
- [ ] Commit `feat: persist alphabet encoding and pronunciation dimensions`.

### Task 6: Integration, documentation and verification

**Files:**
- Modify: `README.md`
- Modify: `docs/ANCIENT_LANGUAGES.md`
- Modify: `docs/UNICODE_ANCIENT_SCRIPTS.md`
- Modify: existing CI workflow where present

- [ ] Document Hebrew, alphabet registry, Unicode/UTF-8 generation and voice controls.
- [ ] Run all ancient-language Python tests plus JSON validation and web syntax checks.
- [ ] Verify registry source/version/checksum metadata and Hebrew UTF-8 values.
- [ ] Verify mixed Hebrew/Greek/Coptic/Aramaic text retains independent script metadata.
- [ ] Commit `docs: document Hebrew Unicode alphabet and voice capabilities`.
- [ ] Perform final branch review before integration.