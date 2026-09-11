# OTA Analysis

FlashTool can inspect Android OTA ZIP packages and CrAU `payload.bin` headers without applying the update.

## Inspection model

- Locate `payload.bin` and `payload_properties.txt` in OTA ZIP packages.
- Validate the `CrAU` magic and safely parse bounded header fields.
- Report payload major version, manifest size and manifest signature size.
- Preserve the package as read-only input.
- Future versions can parse the signed manifest into partition operations and estimate affected slots.

## A/B safety

The updater model is slot-aware. An inactive slot can be analyzed as an update target, but FlashTool must not infer that a slot is safe to activate merely from package presence. AVB and rollback metadata must remain part of preflight.

## Source

The canonical reference is Android Open Source Project `system/update_engine`, including its published payload format and update flow documentation.
