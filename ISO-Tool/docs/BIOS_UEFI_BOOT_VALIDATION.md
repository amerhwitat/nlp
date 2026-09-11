# BIOS / UEFI boot validation

## BIOS first stage

`boot/bios/first_stage.asm` is assembled as a flat 512-byte boot sector. `ORG 0x7C00` models the conventional BIOS load address. The example menu uses `INT 10h` and `INT 16h` for first-stage display/input and carries the `0x55AA` boot signature.

ISO-Tool should validate this artifact with NASM and, when available, boot it only inside QEMU. A successful assembly or `0x55AA` signature is not itself proof that the full operating system boots.

## UEFI

`boot/uefi/entry.c` documents the EFI application contract. A production UEFI image must be built as the required PE/COFF EFI application and installed at the architecture-specific EFI boot path. Firmware selects the image load address and invokes the EFI entry point. BIOS interrupt services are not used by this path.

The value `0x8000` is not a standard UEFI entry address. ISO-Tool allows it only in an explicitly named custom loader/test profile where the loader itself defines that convention.

## Validation sequence

1. Validate menu/profile schema.
2. Verify required artifacts exist.
3. Verify BIOS boot-sector size/signature and declared `0x7C00` contract.
4. Verify UEFI artifact format/path/entry contract when tooling is available.
5. Run QEMU BIOS validation when QEMU is installed.
6. Run QEMU + OVMF UEFI validation when both are installed.
7. Record serial/log/exit evidence.
8. On unavailable or failed optional entry, select its configured fallback and repeat validation.
9. Stop only when a valid eligible entry is found or no fallback remains.

## Evidence levels

- `static`: source/artifact checks only.
- `assembled`: assembler completed successfully.
- `emulated`: QEMU boot test produced the expected evidence.
- `unverified`: required external firmware/tooling is not installed.

ISO-Tool must display the evidence level in its application log and build report.
