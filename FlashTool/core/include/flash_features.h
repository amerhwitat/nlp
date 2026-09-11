#pragma once

#include "flashtool.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum chm_image_kind {
    CHM_IMAGE_UNKNOWN,
    CHM_IMAGE_ANDROID_BOOT,
    CHM_IMAGE_ANDROID_INIT_BOOT,
    CHM_IMAGE_ANDROID_VENDOR_BOOT,
    CHM_IMAGE_ANDROID_DTBO,
    CHM_IMAGE_SPARSE,
    CHM_IMAGE_VBMETA,
    CHM_IMAGE_SUPER,
    CHM_IMAGE_OTA_PAYLOAD,
    CHM_IMAGE_OTA_ZIP
} chm_image_kind_t;

typedef enum chm_slot_state {
    CHM_SLOT_UNKNOWN,
    CHM_SLOT_A,
    CHM_SLOT_B,
    CHM_SLOT_NON_AB
} chm_slot_state_t;

typedef enum chm_preflight_code {
    CHM_PREFLIGHT_OK,
    CHM_PREFLIGHT_INVALID_INPUT,
    CHM_PREFLIGHT_NO_TRANSPORT,
    CHM_PREFLIGHT_LOCKED_WRITE,
    CHM_PREFLIGHT_PARTITION_MISMATCH,
    CHM_PREFLIGHT_IMAGE_TOO_LARGE,
    CHM_PREFLIGHT_ROLLBACK_RISK,
    CHM_PREFLIGHT_UNSUPPORTED
} chm_preflight_code_t;

typedef struct chm_image_info {
    chm_image_kind_t kind;
    uint64_t file_size;
    uint64_t declared_size;
    uint32_t header_version;
    uint32_t flags;
    int has_avb;
    int has_signature;
    int sparse;
} chm_image_info_t;

typedef struct chm_partition_info {
    char name[128];
    uint64_t size_bytes;
    int logical;
    int readonly;
    chm_slot_state_t slot;
} chm_partition_info_t;

typedef struct chm_preflight_result {
    chm_preflight_code_t code;
    int safe_to_execute;
    int requires_confirmation;
    int requires_unlock;
    char reason[256];
} chm_preflight_result_t;

const char* chm_image_kind_name(chm_image_kind_t kind);
const char* chm_preflight_code_name(chm_preflight_code_t code);
const char* chm_slot_state_name(chm_slot_state_t slot);
int chm_preflight(const chm_device_info_t*, const chm_flash_plan_t*, const chm_image_info_t*, const chm_partition_info_t*, chm_preflight_result_t*);

#ifdef __cplusplus
}
#endif
