#include "../core/include/flashtool.h"

int flashtool_safe_plan(const chm_device_info_t *device, const chm_flash_plan_t *plan) {
    return chm_validate_flash_plan(device, plan);
}
