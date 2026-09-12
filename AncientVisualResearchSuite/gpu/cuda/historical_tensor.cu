#include <cuda_runtime.h>

__global__ void avrs_add(const float* a, const float* b, float* out, unsigned n) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

extern "C" void avrs_launch_add(const float* a, const float* b, float* out, unsigned n) {
    avrs_add<<<(n + 255u) / 256u, 256u>>>(a, b, out, n);
}
