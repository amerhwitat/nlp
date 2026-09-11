# Chimera II Mobile FlashKit

A safety-first, GUI-oriented Android device inspection and authorized firmware flashing toolkit.

## Scope

FlashTool provides a common orchestration layer for Android Platform Tools (`adb`/`fastboot`), USB device discovery, partition inspection, image validation, AVB metadata inspection, A/B slot awareness, dry-run planning, and post-flash verification.

It does **not** bypass bootloader security, FRP, credentials, OEM authorization, secure boot, or vendor protections. Bootloader unlocking is exposed only through the device/OEM-supported workflow and requires explicit confirmation.

## Layout

- `core/` — canonical C++ engine and stable C ABI
- `c/` — C integration layer
- `cpp/` — native C++ implementation and GUI integration
- `asm/` — architecture-specific low-level support
- `python/` — automation and inspection API
- `java/` — Java orchestration/API layer
- `backends/` — ADB, Fastboot, Fastbootd and USB transport adapters
- `device/` — device, RAM/storage and partition discovery
- `images/` — image parsing, sparse-image and payload handling
- `security/` — hashes, signatures and AVB inspection
- `chimera/` — Chimera II RegisterN/C8192/R8192 acceleration interfaces
- `gui/` — native GUI architecture
- `tests/` — unit, parser, compatibility and safety tests
- `docs/` — design, support and operational documentation

## Design principle

The C++ core is canonical. Other language editions bind to the same safety contracts instead of maintaining independent flashing implementations.

## Supported host toolchains

The build matrix targets GCC, Clang/LLVM, MSVC/MinGW, Android NDK Clang, OpenJDK/Gradle and supported assemblers. The project does not claim that every compiler ever released is supported.

## Safety

All write operations require device identity and partition compatibility checks. Destructive operations must be explicit. The default workflow is inspect -> validate -> dry-run -> confirm -> flash -> verify.

## Android tooling

Install Android Platform-Tools separately from the official Android developer distribution. Do not commit vendor binaries or proprietary factory images to this repository.
