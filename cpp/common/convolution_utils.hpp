// cpp/common/convolution_utils.hpp
//
// Shared "apply a small kernel to every pixel, per channel" logic, so
// Gaussian blur (FR-05), Sobel (FR-06), and the generic convolution
// engine (FR-07) all use the same border-handling code instead of each
// reimplementing it slightly differently.

#pragma once

#include "image_io.hpp"
#include <vector>

namespace gcv
{

    // Applies a square, odd-sized kernel (e.g. 3x3, 5x5) to `img`, independently
    // per channel. Out-of-bounds neighbor pixels are handled by clamping to the
    // nearest edge pixel (the common choice for image filters — avoids
    // artificially darkening/lightening borders the way zero-padding would).
    //
    // `kernel` must have kernel_size * kernel_size elements, row-major.
    // Output values are clamped to [0, 255] after rounding.
    Image apply_kernel_cpu(const Image &img, const std::vector<float> &kernel,
                           int kernel_size);

    // Same as apply_kernel_cpu, but returns raw float sums with no clamping or
    // rounding. Needed by operations like Sobel (FR-06) where Gx/Gy can go
    // negative or exceed 255 and must stay that way until a later combining
    // step (e.g. magnitude = sqrt(Gx^2 + Gy^2)) decides the final byte value.
    // Output has width * height * channels elements, same layout as Image::data.
    std::vector<float> apply_kernel_raw_cpu(const Image &img,
                                            const std::vector<float> &kernel,
                                            int kernel_size);

} // namespace gcv