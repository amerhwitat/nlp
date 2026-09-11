# Chimera II Mobile FlashTool

FlashTool is a safety-first Android device inspection, image analysis, compatibility and authorized firmware servicing framework.

## Core workflow

```text
inspect -> identify -> analyze -> compatibility preflight -> dry-run
       -> explicit confirmation -> authorized write -> verify -> journal
```

Analysis commands are offline and non-destructive. Write operations must remain behind explicit authorization and device/OEM security checks.

## New capabilities

- ADB, Fastboot and Fastbootd transport model.
- A/B and non-A/B slot model.
- Device capability and partition compatibility preflight.
- Android boot/init_boot/vendor_boot magic recognition.
- Android sparse-image header inspection.
- AVB/VBMeta and AVB-footer detection.
- CrAU `payload.bin` header inspection.
- OTA ZIP discovery for `payload.bin`, `payload_properties.txt`, and Android metadata.
- C/C++/Python/Java parity for safety and compatibility concepts.
- JSON-friendly offline analysis suitable for automation and Web UI integration.
- CMake smoke test and Python unit tests.
- Optional Chimera RegisterN/C8192/R8192 acceleration boundary for future hashing and analysis work.

## Layout

- `core/` — canonical C++ safety and analysis engine
- `c/` — stable C ABI adapter
- `python/` — offline analyzer and automation API
- `java/` — Java API model
- `cli/` — safe offline CLI
- `tests/` — parser, policy and C++ smoke tests
- `docs/` — architecture, security, AVB, OTA, dynamic partition and research documentation
- `build/` — CMake configuration
- `asm/` — reserved for verified low-level primitives

## Safety boundary

FlashTool does **not** bypass FRP, credentials, OEM authorization, secure boot, AVB, rollback protection, or vendor cryptographic controls. It does not ship vendor binaries or proprietary factory images.

A locked device is never treated as writable by the core policy layer. Dry-run and inspection paths remain available for locked devices.

## Host tools

Install Android SDK Platform-Tools separately and use the current supported `adb`/`fastboot` release. FlashTool does not embed or redistribute Platform-Tools binaries.

## Python CLI

From the repository root:

```text
python FlashTool/cli/flashtool_cli.py inspect path/to/artifact
python FlashTool/cli/flashtool_cli.py analyze path/to/artifact --pretty
```

## Testing

Python tests are offline and do not require a phone:

```text
python -m unittest discover -s FlashTool/tests
```

The CMake configuration includes `flashtool_core_smoke` for validating the canonical C++ safety layer.

## Research basis

The enhanced architecture follows public Android documentation and upstream source for AVB, update_engine/CrAU payloads, dynamic partitions and Platform-Tools. See `docs/DEEP_RESEARCH.md` for the research record and links.
