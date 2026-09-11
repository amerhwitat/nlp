#include "../include/flash_features.h"
#include <cstdio>
#include <cstring>

namespace {
void set_reason(chm_preflight_result_t* r, const char* text) {
    if (!r) return;
    std::snprintf(r->reason, sizeof(r->reason), "%s", text ? text : "");
}
}

extern "C" const char* chm_image_kind_name(chm_image_kind_t kind) {
    switch (kind) {
        case CHM_IMAGE_ANDROID_BOOT: return "android-boot";
        case CHM_IMAGE_ANDROID_INIT_BOOT: return "android-init_boot";
        case CHM_IMAGE_ANDROID_VENDOR_BOOT: return "android-vendor_boot";
        case CHM_IMAGE_ANDROID_DTBO: return "android-dtbo";
        case CHM_IMAGE_SPARSE: return "sparse";
        case CHM_IMAGE_VBMETA: return "vbmeta";
        case CHM_IMAGE_SUPER: return "super";
        case CHM_IMAGE_OTA_PAYLOAD: return "ota-payload";
        case CHM_IMAGE_OTA_ZIP: return "ota-zip";
        default: return "unknown";
    }
}

extern "C" const char* chm_preflight_code_name(chm_preflight_code_t code) {
    switch (code) {
        case CHM_PREFLIGHT_OK: return "ok";
        case CHM_PREFLIGHT_INVALID_INPUT: return "invalid-input";
        case CHM_PREFLIGHT_NO_TRANSPORT: return "no-transport";
        case CHM_PREFLIGHT_LOCKED_WRITE: return "locked-write";
        case CHM_PREFLIGHT_PARTITION_MISMATCH: return "partition-mismatch";
        case CHM_PREFLIGHT_IMAGE_TOO_LARGE: return "image-too-large";
        case CHM_PREFLIGHT_ROLLBACK_RISK: return "rollback-risk";
        default: return "unsupported";
    }
}

extern "C" const char* chm_slot_state_name(chm_slot_state_t slot) {
    switch (slot) {
        case CHM_SLOT_A: return "a";
        case CHM_SLOT_B: return "b";
        case CHM_SLOT_NON_AB: return "non-ab";
        default: return "unknown";
    }
}

extern "C" int chm_preflight(const chm_device_info_t* d,
                              const chm_flash_plan_t* p,
                              const chm_image_info_t* image,
                              const chm_partition_info_t* partition,
                              chm_preflight_result_t* out) {
    if (!out) return 0;
    std::memset(out, 0, sizeof(*out));
    out->code = CHM_PREFLIGHT_INVALID_INPUT;
    out->requires_confirmation = 1;

    if (!d || !p || !image || !partition || !p->partition[0] || !p->image_path[0]) {
        set_reason(out, "device, plan, image and partition metadata are required");
        return 0;
    }
    if (d->transport == CHM_TRANSPORT_NONE) {
        out->code = CHM_PREFLIGHT_NO_TRANSPORT;
        set_reason(out, "no supported device transport is connected");
        return 0;
    }
    if (!p->dry_run && !d->bootloader_unlocked) {
        out->code = CHM_PREFLIGHT_LOCKED_WRITE;
        out->requires_unlock = 1;
        set_reason(out, "non-dry-run writes require the device's authorized unlocked state");
        return 0;
    }
    if (std::strcmp(p->partition, partition->name) != 0) {
        out->code = CHM_PREFLIGHT_PARTITION_MISMATCH;
        set_reason(out, "requested partition does not match discovered partition metadata");
        return 0;
    }
    if (image->declared_size && partition->size_bytes && image->declared_size > partition->size_bytes) {
        out->code = CHM_PREFLIGHT_IMAGE_TOO_LARGE;
        set_reason(out, "image declared size exceeds target partition size");
        return 0;
    }
    if (partition->readonly && !p->dry_run) {
        out->code = CHM_PREFLIGHT_UNSUPPORTED;
        set_reason(out, "target partition is read-only");
        return 0;
    }

    out->code = CHM_PREFLIGHT_OK;
    out->safe_to_execute = p->dry_run ? 1 : 0;
    set_reason(out, p->dry_run ? "dry-run passed; no write will occur" : "preflight passed; explicit execution confirmation remains required");
    return 1;
}
