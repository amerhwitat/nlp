# Smart boot, Live and installation architecture

ISO-Tool treats repository documentation as evidence for image organization. An optional LLM/RNN adapter may propose a layout, but deterministic validation and authorization remain mandatory.

## First-stage firmware paths

### BIOS

The BIOS first-stage artifact is a 512-byte real-mode boot sector assembled with NASM. Its conventional BIOS load/handoff address is physical `0x7C00` (`0000:7C00`). The first-stage menu may use BIOS interrupt services such as `INT 10h` for display and `INT 16h` for keyboard input while in real mode. The sample first-stage uses these services only for its initial menu and then hands off to a validated next stage.

### UEFI

UEFI is a different execution environment. It does **not** use BIOS interrupts and does not define a universal `0x8000` entry address. A UEFI boot target is normally a PE/COFF EFI application loaded by firmware and entered through its EFI image entry point. ISO-Tool therefore records `efi_main`/the selected EFI entry contract and leaves the actual image load address to firmware.

`0x8000` is supported only as an explicitly configured address for a custom loader/test profile. It must never be described as the generic UEFI entry address.

## Boot validation and fallback

Before final boot-image publication, ISO-Tool statically checks the configured boot artifacts. When QEMU is installed, it can generate an isolated BIOS or UEFI test invocation; the test result is recorded rather than treating static inspection as proof of successful boot.

The menu contains an ordered fallback chain. If the preferred entry is missing, invalid, or fails an isolated validation test, the validator records the reason and selects the next eligible entry. This applies to optional entries such as GRUB, Windows Boot Manager, LILO and Syslinux. A required boot path that has no valid fallback remains a final build failure.

The runtime application displays each attempt and fallback decision in its live operation-details pane.

## Generated menu

The generated image can contain coordinated entries for native Chimera II, GRUB 2, optional LILO/Syslinux legacy paths, Linux EFI loaders, and Windows Boot Manager chainloading. Loader artifacts are included only when present and compatible with the selected profile.

UEFI removable media uses the architecture-specific `EFI/BOOT/BOOT{machine-type}.EFI` convention. Windows Boot Manager binaries are treated as externally supplied Windows artifacts rather than recreated by ISO-Tool.

## Live media

A Live profile places the kernel/initramfs and runtime filesystem in the image. The install profile is a separate menu target that writes a selected OS image to an explicitly selected target.

## Installation safety

Writing an OS image to HDD/SSD/NVMe is destructive. ISO-Tool must display the selected target and require explicit confirmation. The core application never silently chooses a physical disk.

## Offline recovery

Local source repositories can be analyzed and built without Internet. Remote acquisition can monitor connectivity and wait for restoration before retrying network operations. Network state and retry activity are visible in the live log.

## Fail-forward

Recoverable failures in optional or independent boot/image jobs are logged, recorded, and followed by the next job. Fatal authorization, safety, staging, required-boot, and final-integrity failures are not silently bypassed.
