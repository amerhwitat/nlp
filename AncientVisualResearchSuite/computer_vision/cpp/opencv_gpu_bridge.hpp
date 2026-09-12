#pragma once
#include <opencv2/core.hpp>

namespace avrs::vision {
// Uploads an image to the best OpenCV-supported accelerator when available.
// The returned matrix remains valid on CPU-only builds.
cv::Mat preprocessHistoricalImage(const cv::Mat& input, double scale = 1.0);
}
