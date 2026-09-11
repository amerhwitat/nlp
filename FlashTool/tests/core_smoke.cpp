#include "../core/include/flashtool.h"
#include "../core/include/flash_features.h"
#include <cassert>
#include <cstring>

int main() {
    chm_device_info_t device{};
    device.transport = CHM_TRANSPORT_FASTBOOT;
    device.bootloader_unlocked = 0;

    chm_flash_plan_t plan{};
    std::strcpy(plan.partition, "boot");
    std::strcpy(plan.image_path, "boot.img");
    plan.dry_run = 1;
    plan.verify = 1;

    chm_image_info_t image{};
    image.kind = CHM_IMAGE_ANDROID_BOOT;
    image.declared_size = 1024;

    chm_partition_info_t partition{};
    std::strcpy(partition.name, "boot");
    partition.size_bytes = 4096;

    chm_preflight_result_t result{};
    assert(chm_preflight(&device, &plan, &image, &partition, &result) == 1);
    assert(result.code == CHM_PREFLIGHT_OK);

    plan.dry_run = 0;
    assert(chm_preflight(&device, &plan, &image, &partition, &result) == 0);
    assert(result.code == CHM_PREFLIGHT_LOCKED_WRITE);

    return 0;
}
