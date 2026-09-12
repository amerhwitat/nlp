#include "avrs/gpu_backend.hpp"
#include <algorithm>

namespace avrs::gpu {

std::vector<DeviceInfo> enumerate() {
    // Runtime adapters are discovered by optional backend modules. CPU is always valid.
    return {{"Portable CPU fallback", Backend::CPU, 0, true}};
}

Backend selectBest() {
    const auto devices = enumerate();
    return std::max_element(devices.begin(), devices.end(), [](const auto& a, const auto& b) {
        return static_cast<int>(a.backend) < static_cast<int>(b.backend);
    })->backend;
}
}
