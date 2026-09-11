/* ISO-Tool minimal UEFI application entry contract.
 * UEFI is NOT entered through BIOS interrupts and has no universal 0x8000 entry address.
 * Firmware loads this PE/COFF EFI application and calls efi_main().
 */
#include <stdint.h>

typedef void *EFI_HANDLE;
typedef uint64_t EFI_STATUS;
typedef struct EFI_SYSTEM_TABLE EFI_SYSTEM_TABLE;

#define EFI_SUCCESS ((EFI_STATUS)0)

EFI_STATUS efi_main(EFI_HANDLE image, EFI_SYSTEM_TABLE *system_table) {
    (void)image;
    (void)system_table;
    /* Real implementation supplies the generated boot menu and handoff policy. */
    return EFI_SUCCESS;
}
