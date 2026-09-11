# Boot image integration

ISO-Tool treats boot firmware targets as explicit image components rather than guessing them.

Supported integration profiles:

- BIOS/legacy MBR boot sector
- GPT disk layout
- UEFI EFI System Partition and `.efi` loader
- El Torito optical boot catalog
- BIOS + UEFI hybrid boot
- ISO/IMG hybrid layouts

Boot binaries are compiled/assembled first, then injected into the image through the selected image backend. MASM (`ml`/`ml64`) and NASM are discovered locally. The tool never fabricates a boot sector: the user selects an existing boot entry source or a documented generated template.
