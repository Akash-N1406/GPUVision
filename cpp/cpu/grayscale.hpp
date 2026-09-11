// cpp/cpu/grayscale.hpp
//
// FR-03: RGB -> Grayscale conversion, CPU baseline.
// Gray = 0.299*R + 0.587*G + 0.114*B (per SRS section 8).

#pragma once

#include "../common/image_io.hpp"

namespace gcv
{

    // Converts an RGB Image (channels == 3) to a single-channel grayscale Image.
    // Every pixel is processed sequentially on the CPU — this is intentionally
    // the naive, one-thread version so it's a fair baseline for the CUDA
    // "1 thread = 1 pixel" kernel in Phase 3.
    Image grayscale_cpu(const Image &rgb);

} // namespace gcv