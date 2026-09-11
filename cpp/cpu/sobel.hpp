// cpp/cpu/sobel.hpp
//
// FR-06: Sobel edge detection, CPU baseline.
// Operates on grayscale — converts internally if given an RGB image.
// Gx, Gy per SRS section 11; magnitude G = sqrt(Gx^2 + Gy^2).

#pragma once

#include "../common/image_io.hpp"

namespace gcv
{

    // Returns a single-channel edge-magnitude image, same width/height as the
    // input. If `img` is RGB it's converted to grayscale first (via
    // grayscale_cpu); if it's already single-channel it's used as-is.
    Image sobel_cpu(const Image &img);

} // namespace gcv