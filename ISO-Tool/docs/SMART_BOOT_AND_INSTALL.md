# Smart boot, Live and installation architecture

ISO-Tool now treats the downloaded repository as evidence for image organization. Documentation is indexed for boot/install terminology and produces a machine-readable insight report. An optional LLM/RNN adapter can consume that report and propose a layout; the model is advisory and cannot execute commands or copy arbitrary files without the build-plan authorization layer.

## Boot strategy

The generated image can contain a coordinated menu with entries for a native Chimera II loader, GRUB 2, optional LILO/Syslinux legacy paths, Linux EFI loaders, and an optional Windows Boot Manager chainload target. Loader artifacts are included only when present and compatible with the selected image profile.

UEFI removable media has a standardized `EFI/BOOT/BOOTx64.EFI` convention (with architecture-specific names for other processor types), so the image planner always reserves the appropriate EFI boot path. citeturn0search48turn0search5

Windows boot environments use Boot Manager/BCD structures; ISO-Tool treats those as Windows-provided artifacts rather than recreating Microsoft's binaries. BCDBoot can provision BIOS, UEFI, or both when operating on a Windows system image. citeturn0search2turn0search8

## Live media

A Live profile places a kernel/initramfs and runtime filesystem in the image and starts the OS without installing it. The installer profile is a separate menu target that runs the trusted installer and writes a selected OS image to an explicitly selected target disk.

## Installation safety

Writing an OS image to HDD/SSD/NVMe is destructive. ISO-Tool must display the target device, size, model and destructive-operation warning and require explicit confirmation. The core application never chooses a physical disk automatically.

## LLM/RNN boundary

The smart module may classify documentation, infer likely boot artifacts, suggest a directory layout, identify build commands from recognized metadata, and rank loader candidates. It must not invent or execute commands, modify firmware/NVRAM, bypass signatures, or select a physical installation target. All actions pass through a deterministic plan validator.
