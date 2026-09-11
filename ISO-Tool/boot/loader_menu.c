/* Minimal deterministic boot-menu source template.
 * Actual BIOS/UEFI handoff remains loader-specific; ISO-Tool generates the
 * menu data and invokes the selected trusted loader backend rather than
 * embedding incompatible proprietary boot code. */
#include <stdint.h>
struct iso_boot_entry { const char *id; const char *label; const char *loader; uint32_t flags; };
static const struct iso_boot_entry entries[] = {
 {"chimera2-live","Chimera II OS - Live","chimera2",1},
 {"chimera2-install","Chimera II OS - Install","chimera2",2},
 {"grub2","GRUB 2 / Linux","grub2",4},
 {"windows","Windows Boot Environment","windows-bootmgr",8},
 {"lilo","Legacy Linux / LILO","lilo",16},
 {"syslinux","Syslinux / Legacy","syslinux",32}
};
uint32_t iso_boot_entry_count(void) { return (uint32_t)(sizeof(entries)/sizeof(entries[0])); }
