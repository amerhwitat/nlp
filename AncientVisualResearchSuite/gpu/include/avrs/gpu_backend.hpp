#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace avrs::gpu {

enum class Backend { Auto, CPU, OpenGL, DirectX12, CUDA, OpenCV, Unreal, Unity };
struct DeviceInfo { std::string name; Backend backend{Backend::CPU}; uint64_t memoryBytes{}; bool compute{}; };
struct Buffer { std::vector<float> values; };

class BackendAdapter {
public:
    virtual ~BackendAdapter() = default;
    virtual Backend backend() const noexcept = 0;
    virtual bool available() const noexcept = 0;
    virtual DeviceInfo device() const = 0;
    virtual void add(const Buffer& a, const Buffer& b, Buffer& out) const = 0;
};

std::vector<DeviceInfo> enumerate();
Backend selectBest();
}
