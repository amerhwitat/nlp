#include "../include/flashtool.h"
#include <cstring>

extern "C" int chm_validate_flash_plan(const chm_device_info_t* d, const chm_flash_plan_t* p) {
    if (!d || !p || !p->partition[0] || !p->image_path[0]) return 0;
    if (d->transport == CHM_TRANSPORT_NONE) return 0;
    /* Never silently write to a locked device. Actual policy is delegated to
       the transport/OEM backend; this core function only gates obvious risk. */
    if (!d->bootloader_unlocked && !p->dry_run) return 0;
    return 1;
}

extern "C" int chm_plan_requires_unlock(const chm_device_info_t* d, const char* partition) {
    if (!d || !partition) return 1;
    if (!d->bootloader_unlocked) return 1;
    return 0;
}

extern "C" const char* chm_transport_name(chm_transport_t t) {
    switch (t) {
        case CHM_TRANSPORT_ADB: return "ADB";
        case CHM_TRANSPORT_FASTBOOT: return "Fastboot";
        case CHM_TRANSPORT_FASTBOOTD: return "Fastbootd";
        default: return "None";
    }
}
