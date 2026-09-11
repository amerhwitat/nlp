# FlashTool Enhancement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand FlashTool into a safety-first Android device, image, OTA, partition, verification, and compatibility analysis framework while preserving the existing C++ safety boundary.

**Architecture:** Keep the C++ core canonical for device/flash policy and introduce portable analyzers for Android boot images, sparse images, AVB metadata, dynamic partitions, and CrAU OTA payloads. Python provides a richer offline analyzer/CLI, Java and C expose the same data model, and tests validate policy and parsers without requiring a physical device.

**Tech Stack:** C++20/C11, Python 3.8+, Java 17+, CMake 3.20+, unittest, GitHub Actions; optional external tools such as adb/fastboot remain host dependencies rather than embedded binaries.

**Spec:** `FlashTool/docs/ENHANCED_ARCHITECTURE.md`

## Global Constraints

- Never bypass FRP, credentials, OEM authorization, secure boot, AVB, rollback protection, or bootloader security.
- Default workflow remains inspect -> validate -> dry-run -> explicit confirmation -> execute -> verify.
- C++ remains the canonical safety/policy layer.
- Offline image analysis must not require a connected device.
- GUI, CLI, C, Python, and Java APIs must describe the same compatibility and safety state.
- Do not commit vendor binaries, proprietary firmware, device secrets, or signing keys.

---

### Task 1: Core analysis data model

**Files:**
- Create: `FlashTool/core/include/flash_features.h`
- Create: `FlashTool/core/src/flash_features.cpp`

- [x] Add transport/device capability, image-kind, slot-state, compatibility, and analysis-result enums/records.
- [x] Add deterministic validation functions that reject missing identity, unsupported transport, locked write attempts, and incompatible image sizes.
- [x] Add human-readable enum names.
- [x] Keep all operations non-destructive until the caller explicitly requests execution.

### Task 2: Python offline analyzer

**Files:**
- Create: `FlashTool/python/__init__.py`
- Create: `FlashTool/python/analyzer.py`
- Modify: `FlashTool/python/flashtool.py`

- [x] Detect Android boot-family magic.
- [x] Detect Android sparse image headers and checksum field.
- [x] Parse CrAU OTA payload header and expose manifest/signature offsets without applying it.
- [x] Inspect ZIP OTA packages for payload.bin, payload_properties.txt and Android metadata.
- [x] Inspect AVB footer/VBMeta magic and expose bounded header fields including rollback metadata.
- [x] Produce JSON-safe analysis reports and compatibility preflight results.

### Task 3: CLI

**Files:**
- Create: `FlashTool/cli/flashtool_cli.py`

- [x] Add `inspect`, `analyze`, `preflight`, and `dry-run` commands.
- [x] Never execute a destructive operation from analysis commands.
- [x] Support JSON output for automation.

### Task 4: C and Java parity

**Files:**
- Modify: `FlashTool/c/flashtool.c`
- Modify: `FlashTool/java/src/main/java/org/chimera/flashtool/FlashTool.java`

- [x] Expose analysis/compatibility structures through the C ABI.
- [x] Mirror core analysis models in Java.
- [x] Preserve existing validation behavior.

### Task 5: Documentation and research record

**Files:**
- Create: `FlashTool/docs/ENHANCED_ARCHITECTURE.md`
- Create: `FlashTool/docs/AVB_ANALYSIS.md`
- Create: `FlashTool/docs/OTA_ANALYSIS.md`
- Create: `FlashTool/docs/DYNAMIC_PARTITIONS.md`
- Create: `FlashTool/docs/PLATFORM_TOOLS.md`
- Create: `FlashTool/docs/DEEP_RESEARCH.md`
- Modify: `FlashTool/README.md`
- Modify: `FlashTool/markdown.md`

- [x] Document Android Verified Boot, rollback, A/B, dynamic partitions, CrAU payloads, Platform-Tools, archive handling, and safety boundaries.
- [x] Record official sources and implementation decisions.

### Task 6: Tests and build integration

**Files:**
- Modify: `FlashTool/tests/test_flashtool.py`
- Create: `FlashTool/tests/test_analyzer.py`
- Modify: `FlashTool/build/CMakeLists.txt`
- Create: `FlashTool/tests/core_smoke.cpp`
- Create: `FlashTool/tests/fixtures/README.md`
- Create: `.github/workflows/flashtool.yml`

- [x] Add policy tests for locked devices, no transport and dry-run behavior.
- [x] Add parser tests for Android magic, sparse images, CrAU headers, ZIP OTA discovery and AVB fields.
- [x] Keep tests deterministic and offline.
- [x] Verify C++ build configuration with GitHub Actions; the C++ smoke job passed on commit `a135a24a806bfa4ef94e3671aa665f47bc7aed61`.
- [x] Reproduce and fix the Python package-path and sparse-header test issues locally; latest Python test mirror runs cleanly with 6 tests passing.
