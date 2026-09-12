# Ancient-language pronunciation and voice controls

The ancient-language workbench separates the text being displayed from its pronunciation representation.

## Modes

- **Modern** — a modern language voice, such as Modern Hebrew.
- **Scholarly** — a convention-driven pronunciation supplied by a documented scholarly profile.
- **Reconstructed** — a historical reconstruction. It is never presented as a guaranteed recording of ancient speech.
- **Reference** — phoneme-aware or recorded reference pronunciation with source metadata.

## UI

The transliteration and translation areas each expose:

- Listen
- Pause
- Resume
- Stop
- Replay
- voice selection
- locale selection
- rate
- pitch
- volume
- pronunciation mode

The browser Web Speech API is the baseline client capability. If it is unavailable, the UI reports that speech synthesis is unavailable instead of silently claiming that an ancient voice exists.

## Data provenance

Every pronunciation profile stores language stage, dialect where applicable, transliteration system, pronunciation type, locale, IPA/phoneme information where available, backend, confidence, source and model/version. Audio assets also store rights and checksum metadata.

## Unicode and alphabet data

Alphabet/sign tables are structured data. The offline registry stores Unicode code points and deterministic UTF-8 byte sequences. A generator can rebuild the complete Unicode snapshot from a pinned UCD release. This allows the application to cover scripts that are not conventional alphabets, including cuneiform and Egyptian hieroglyphs, without incorrectly forcing them into an alphabet model.
