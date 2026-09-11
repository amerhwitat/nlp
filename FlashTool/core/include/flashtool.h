#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef enum chm_transport { CHM_TRANSPORT_NONE, CHM_TRANSPORT_ADB, CHM_TRANSPORT_FASTBOOT, CHM_TRANSPORT_FASTBOOTD } chm_transport_t;
typedef struct chm_device_info {
    char serial[128];
    char product[128];
    char state[32];
    uint64_t ram_bytes;
    uint64_t storage_bytes;
    int bootloader_unlocked;
    int slot_count;
    chm_transport_t transport;
} chm_device_info_t;
typedef struct chm_flash_plan {
    char partition[128];
    char image_path[512];
    int dry_run;
    int verify;
} chm_flash_plan_t;

int chm_validate_flash_plan(const chm_device_info_t*, const chm_flash_plan_t*);
int chm_plan_requires_unlock(const chm_device_info_t*, const char* partition);
const char* chm_transport_name(chm_transport_t);

#ifdef __cplusplus
}
#endif
