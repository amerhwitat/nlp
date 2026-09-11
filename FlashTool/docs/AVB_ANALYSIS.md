# AVB Analysis

FlashTool treats Android Verified Boot (AVB) as a security boundary rather than something to disable.

## Planned inspection

- Detect VBMeta structures and AVB footer markers.
- Report hash and hashtree descriptors when a complete AVB parser is available.
- Record chained VBMeta relationships.
- Surface rollback-index metadata for compatibility review.
- Distinguish locked, unlocked and unknown verification state.
- Compare expected image metadata without rewriting signatures.

## Policy

A mismatch is reported as a compatibility/security finding. FlashTool does not generate bypass instructions, remove verification metadata, defeat rollback protection, or alter OEM trust roots.

## Source

Android Verified Boot 2.0 documentation and upstream `external/avb` are the reference implementation and specification sources.
