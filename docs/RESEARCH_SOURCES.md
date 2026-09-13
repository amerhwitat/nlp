# Research and interoperability sources

The implementation was reviewed against current public documentation and research resources for ancient-language corpora, Unicode, transliteration, and browser voice APIs.

## Ancient North Arabian

**OCIANA — Online Corpus of the Inscriptions of Ancient North Arabia** provides readings in Roman transliteration, English translations, references, commentary, bibliography, provenance, relationships, and images/facsimiles. It is a primary provenance target for Ancient North Arabian workflows.

https://ociana.osu.edu/

## Cuneiform / Akkadian / Sumerian

**ORACC** documents ATF conventions, language/dialect metadata, Unicode transliteration, and JSON text editions containing structured transliteration and lemmatization. This supports a future import/provider layer that preserves object → surface → column → line → word → sign hierarchy.

https://build-oracc.museum.upenn.edu/doc/help/editinginatf/primer/index.html

https://oracc.museum.upenn.edu/compass/downloads/2_3_Data_Acquisition_ORACC.html

**CDLI Machine Translation** provides an open-source Sumerian/English machine-translation research implementation and is a candidate provider/dataset integration target rather than a claim of built-in translation coverage.

https://github.com/cdli-gh/Machine-Translation

## Unicode

Unicode script charts and code charts are the authoritative baseline for encoded script/character identity. The application therefore separates Unicode identification from language translation and phonetic reconstruction.

https://www.unicode.org/charts/

https://www.unicode.org/charts/script/index_list.html

https://www.unicode.org/standard/supported.html

## Browser voice

The Web Speech API provides SpeechSynthesis and SpeechRecognition. Recognition support is browser-dependent and may use remote processing unless on-device recognition is available and selected. The implementation therefore treats voice as a capability layer and keeps authenticated ancient pronunciation provider-dependent.

https://developer.mozilla.org/en-US/docs/Web/API/Web_Speech_API

https://developer.mozilla.org/en-US/docs/Web/API/SpeechRecognition

## Implementation policy

1. Unicode identity is not treated as translation.
2. Transliteration is kept distinct from translation.
3. Corpus retrieval is distinguished from machine translation.
4. Approximate historical dating is explicitly labeled as approximate.
5. Native ancient pronunciation is never inferred from a modern-language TTS voice.
6. Every universal translation request can be logged with provenance and a deterministic integrity hash.
7. External datasets are integrated through adapters/providers so their licensing and provenance can remain explicit.
