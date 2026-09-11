// cpp/cpu/gaussian_blur.hpp
//
// FR-05: Gaussian blur, CPU baseline.
// Uses a proper normalized Gaussian kernel (not just the SRS's illustrative
// 3x3 [1 2 1; 2 4 2; 1 2 1]/16 example) so blur strength is controllable via
// sigma — useful later when benchmarking different kernel sizes (FR-14).

#pragma once

#include "../common/image_io.hpp"

namespace gcv
{

    // Applies Gaussian blur with the given odd kernel_size (e.g. 3, 5, 7) and
    // standard deviation sigma. Works on RGB or grayscale images.
    Image gaussian_blur_cpu(const Image &img, int kernel_size = 5, float sigma = 1.4f);

} // namespace gcv