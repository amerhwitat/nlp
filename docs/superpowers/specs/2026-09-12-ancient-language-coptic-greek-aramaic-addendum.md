# Ancient Language Intelligence Expansion — Coptic, Greek, and Aramaic Addendum

**Status:** Approved extension to `docs/superpowers/specs/2026-09-12-ancient-language-intelligence-expansion.md`.

## Scope extension

The registry and recognition engine must additionally model Coptic, the historical continuum of Greek, and Aramaic from ancient through modern descendants. Unicode is a script/character encoding authority, not a language catalog; therefore language, historical stage, dialect, script, and writing-system relationships remain separate entities.

## Coptic

Register Coptic as its own script and language family/stage layer, including Sahidic, Bohairic, Fayumic, Akhmimic, Lycopolitan and other attested varieties when supported by authoritative datasets. Preserve mixed Greek/Coptic passages as separate language/script spans. Store Coptic characters using the dedicated Coptic repertoire and retain source-era encodings when encountered for historical corpora. The implementation must not collapse Coptic characters into Greek merely because the scripts share historical ancestry.

## Greek historical continuum

Model Greek as a language family with explicit stages and dialect/orthographic metadata rather than one undifferentiated language. Registry coverage includes Mycenaean/Linear B where linguistically applicable, Archaic Greek, Classical Greek, major epigraphic/dialectal traditions, Hellenistic/Koine Greek, Roman-period Greek, Byzantine/Medieval Greek, and Modern Greek. Store polytonic and monotonic orthographies separately from language-stage identity, and preserve historic letters and numeral characters. Greek transliteration profiles must be versioned by scholarly convention.

## Aramaic historical-to-modern continuum

Register Ancient/Old Aramaic, Imperial Aramaic, Biblical/Official Aramaic, Achaemenid-period varieties, Nabataean and Palmyrene-related Aramaic contexts, Hatran, Syriac traditions, Mandaic, Samaritan Aramaic, and modern Eastern/Western Neo-Aramaic branches where authoritative language metadata is available. Model Syriac and Mandaic as scripts/literary traditions independently from the language-stage relationship. Preserve right-to-left direction, cursive/shaping requirements, vocalization marks, and historical script variants.

## Recognition changes

The hierarchical classifier becomes:

```text
visual encoder
 -> writing-system family
 -> script
 -> historical script stage
 -> language
 -> historical stage / dialect
 -> orthography / direction / layout
 -> glyph or character recognizer
 -> Unicode + scholarly identifier
 -> transliteration profile
 -> morphology / lexicon
 -> contextual reasoning
 -> translation
```

Mixed-script documents must support span-level classification so Greek, Coptic, Aramaic/Syriac and other languages can coexist on one object or page.

## Unicode model

Registry records must store Unicode Script property, block/range references, code points, UTF-8 bytes, normalization behavior, bidi class, combining behavior, and ISO 15924 script code where available. Greek Extended and combining-mark sequences must be represented without losing canonical normalization information. Coptic must use its dedicated script identity. Aramaic-derived RTL scripts must retain shaping and bidi metadata. Unicode 18.0 data must be versioned and imported through a reproducible data-ingestion process rather than hard-coded.

## ER additions

```text
LANGUAGE 1---N LANGUAGE_STAGE
LANGUAGE_STAGE 1---N DIALECT
LANGUAGE_STAGE N---N SCRIPT
SCRIPT 1---N SCRIPT_VARIANT
LANGUAGE_STAGE N---N ORTHOGRAPHY
LANGUAGE_STAGE N---N MODERN_CONTINUATION
LANGUAGE_STAGE 1---N LANGUAGE_ATTESTATION
SCRIPT 1---N TRANSLITERATION_SYSTEM
TRANSLITERATION_SYSTEM 1---N TRANSLITERATION_RULE
```

Each relationship carries source, evidence class, confidence, validity period, and authority identifiers where applicable.

## MADM additions

Add dimensions for language stage, dialect, orthography, script variant, transliteration system, modern continuation, and Unicode normalization profile. Add facts for stage classification, mixed-script spans, transliteration confidence, historical/modern language classification, and orthographic normalization.

## Validation additions

Tests must verify:

- Coptic/Greek character and script identity are not collapsed.
- Polytonic Greek round-trips through Unicode normalization without losing source representation.
- Greek historical stages can coexist with dialect metadata.
- Ancient and modern Aramaic stages remain distinguishable while sharing family relationships.
- Syriac and Mandaic RTL/shaping metadata are preserved.
- Mixed Greek/Coptic/Aramaic spans receive independent script/language hypotheses.
- UTF-8 byte sequences are reproducible from Unicode code points.
- Language-stage-to-script and modern-continuation relationships are provenance-backed.

## Documentation requirements

Update the main README, ancient-language registry documentation, OCR/AI architecture documentation, ER/MADM documentation, Unicode ingestion documentation, API documentation, and test documentation to describe Coptic, Greek historical stages, Aramaic historical/modern stages, Syriac, Mandaic, and mixed-script recognition.

## Research basis

Unicode currently lists Coptic, Greek, Imperial Aramaic, Mandaic, Syriac, Nabataean, Palmyrene, Hatran, Old North Arabian, Old South Arabian and related scripts among its supported scripts, while Unicode's writing-system documentation classifies Coptic and Greek as alphabets and Imperial Aramaic, Syriac, Mandaic and related scripts as abjads. Unicode also documents Coptic as a distinct script from Greek and provides Greek Extended for polytonic representation. The implementation should track the exact Unicode data release used by each generated registry snapshot.
