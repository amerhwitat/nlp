# Hebrew, Alphabet/UTF-8 and Pronunciation Addendum

**Status:** Approved for implementation.

## Hebrew

Add Ancient/Biblical Hebrew, Second Temple/historical Hebrew, Medieval Hebrew, Masoretic Hebrew, and Modern Hebrew as distinct language stages. Preserve Paleo-Hebrew as a historical script context related to Phoenician rather than collapsing it into the modern Hebrew Unicode block. Support Samaritan Hebrew as a related but distinct tradition when authoritative data is available.

## Alphabet and sign registry

Add a versioned registry covering alphabetic characters, combining marks, historic variants, and non-alphabetic sign inventories. Each entry records language/stage, script, character/sign, Unicode code point, Unicode name, UTF-8 bytes, normalization behavior, direction, transliteration, pronunciation metadata, source/version/checksum and validation state.

Unicode UCD/code charts are the normative encoding source. ISO 15924 is used for script codes. Scholarly sources are authority-qualified for historical names, transliteration and pronunciation. The application ships a generated offline snapshot plus the deterministic generator used to refresh it.

## Speech

Add pronunciation profiles to transliterations and translations. Profiles include IPA where available, grapheme-to-phoneme mapping, syllable/stress data, pronunciation type, historical-status label, locale, backend, confidence and provenance.

The web UI adds Listen controls immediately below transliteration and translation text areas. Controls include play, pause, resume, stop, replay, voice, locale, rate, pitch, volume, automatic playback and pronunciation mode. Browser Web Speech API is the baseline; local/server TTS and recorded reference audio are optional adapters. Ancient reconstructed pronunciations must be labeled as reconstructed/reference rather than historically authentic native speech.

## ER/MADM

Add `ALPHABET`, `ALPHABET_MEMBER`, `UTF8_ENCODING`, `PRONUNCIATION_PROFILE`, `PRONUNCIATION_VARIANT`, `VOICE_PROFILE`, and `AUDIO_ASSET`. Add analytical dimensions/facts for alphabet membership, encoding, pronunciation, speech backend and pronunciation confidence.

## Validation

Verify Hebrew RTL/combining behavior, final letters, niqqud/cantillation round trips, Unicode-to-UTF-8 determinism, alphabet membership, provenance, mixed-script handling, Web Speech feature detection, complete voice-control state transitions, and explicit historical-pronunciation uncertainty.