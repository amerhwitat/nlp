#include "opencv_gpu_bridge.hpp"
#include <opencv2/imgproc.hpp>
#if __has_include(<opencv2/core/cuda.hpp>)
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#endif

namespace avrs::vision {
cv::Mat preprocessHistoricalImage(const cv::Mat& input, double scale) {
    if (input.empty()) return {};
    cv::Mat out;
#if __has_include(<opencv2/core/cuda.hpp>)
    if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
        cv::cuda::GpuMat gpu, resized;
        gpu.upload(input);
        cv::cuda::resize(gpu, resized, cv::Size(), scale, scale);
        resized.download(out);
        return out;
    }
#endif
    cv::resize(input, out, cv::Size(), scale, scale, cv::INTER_AREA);
    return out;
}
}
