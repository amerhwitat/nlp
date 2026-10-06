# Aurora Ancient Language Auto OCR

The public Aurora portal exposes `apps/chimera-ii-os/web/ancient-ocr.html` as a browser adapter for this repository's ancient-language research stack.

## Pipeline

`image -> browser OCR / optional ThamudicScan /scan_file -> Unicode/script detection -> ancient-language registry -> transliteration/translation provider -> evidence report`

The browser catalog is synchronized from `ThamudicEpiPlatform/data/schemas/language_registry.json` at portal build time.

## Targets

The documented target catalog is BCP-47/CLDR based. Defaults are `en`, `ar`, `zh`, `ja`; documented examples are `de`, `es`, `fr`, `it`, `ko`, `pt`, `ru`, `tr`, `fa`, `ur`, `hi`, `id`.

## Evidence boundary

Tesseract.js has trained models for some historical languages/scripts (including Ancient Greek `grc`), but not every ancient script. Unsupported image scripts must be routed to a configured research backend or dedicated trained model. Unicode detection identifies encoded script ranges; it does not by itself prove language identity. Translation capability does not imply that a model/corpus is installed.

The portal preserves provider, confidence, provenance and unavailable/provider-required states instead of inventing readings.
