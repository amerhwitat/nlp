# FlashTool Deep Research Record

Date: 2026-09-11

## Findings

### Android Verified Boot

AOSP's AVB documentation describes VBMeta as the signed root metadata object, with hash descriptors, hashtree descriptors, chained partitions, A/B support and rollback protection. FlashTool therefore treats AVB metadata as compatibility/security evidence and never as a bypass target.

Reference: https://android.googlesource.com/platform/external/avb/

### OTA / update_engine

AOSP `update_engine` documents the CrAU payload format: `CrAU` magic, payload version, manifest size, manifest signature size, manifest and payload signatures. The update flow verifies signed metadata before applying operations and uses A/B inactive-slot semantics.

References:
- https://android.googlesource.com/platform/system/update_engine/
- https://android.googlesource.com/platform/system/update_engine/+/master/scripts/update_device.py

### Dynamic partitions

AOSP dynamic partition tooling exposes the `super` metadata model through liblp/partition tools. Sparse `super` images can be converted for analysis, and logical partitions belong to updateable groups with capacity constraints.

Reference: https://android.googlesource.com/platform/system/extras/+/master/partition_tools/README.md

### Platform-Tools

Android Developers' Platform-Tools release notes document `adb` and `fastboot` as the primary SDK tools for device interaction and show that USB and mDNS implementations evolve between releases. FlashTool therefore uses an adapter boundary and capability detection rather than assuming one USB implementation.

Reference: https://developer.android.com/tools/releases/platform-tools

### Archive handling

libarchive is a mature multi-format streaming archive library with automatic format detection. It is a candidate optional dependency for future package inspection, while the current Python analyzer uses the standard library ZIP reader to keep the baseline dependency-free.

Reference: https://github.com/libarchive/libarchive

## New feature decisions

1. Keep C++ policy canonical.
2. Add offline analysis before adding write execution.
3. Model A/B, Fastbootd and dynamic partitions explicitly.
4. Add machine-readable preflight results.
5. Add deterministic parser tests and malformed-input coverage.
6. Keep vendor binaries and proprietary protocols outside the repository.
7. Preserve AVB, rollback, OEM authorization and secure boot as security boundaries.
8. Leave room for Chimera parallel hashing/image analysis without making Chimera hardware mandatory.

## Future research backlog

- Full AVB descriptor parser using upstream-compatible structures.
- Full liblp metadata parser.
- CrAU protobuf manifest decoding and operation graph.
- Android boot image v3/v4/v5 family metadata parsing.
- Virtual A/B snapshot metadata analysis.
- EROFS/ext4 filesystem metadata inspection.
- SBOM/provenance generation.
- Parser fuzzing corpus derived from public test fixtures.
