# UEFI entry

UEFI boot is not a BIOS-interrupt environment. A UEFI firmware loads an architecture-appropriate PE/COFF EFI application and invokes its image entry point with an image handle and system table.

There is no universal UEFI entry address of `0x8000`. ISO-Tool therefore does not hard-code `0x8000` for UEFI. A custom loader may explicitly define `0x8000` as its own test/staging convention, but that value belongs to the custom loader contract.

The minimal `entry.c` documents the `efi_main()` contract. A production build must use the platform's UEFI headers/toolchain and produce the required EFI executable format.
