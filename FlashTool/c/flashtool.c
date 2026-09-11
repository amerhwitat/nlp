#include "../core/include/flashtool.h"
#include "../core/include/flash_features.h"

int flashtool_safe_plan(const chm_device_info_t *device, const chm_flash_plan_t *plan) {
    return chm_validate_flash_plan(device, plan);
}

int flashtool_preflight(const chm_device_info_t *device,
                        const chm_flash_plan_t *plan,
                        const chm_image_info_t *image,
                        const chm_partition_info_t *partition,
                        chm_preflight_result_t *result) {
    return chm_preflight(device, plan, image, partition, result);
}
