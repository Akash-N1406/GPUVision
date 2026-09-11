// cpp/cpu/histogram.hpp
//
// FR-11: histogram computation, CPU baseline.
// Counts pixel-intensity frequencies for a grayscale image, 256 bins
// (0-255). This is the CPU version of the "memory contention" workload —
// on the CPU there's no contention since one thread visits every pixel
// sequentially, but the CUDA version (Phase 4) will have many threads
// incrementing shared bins concurrently, which is exactly why FR-11 exists
// as its own feature rather than folding into grayscale.

#pragma once

#include "../common/image_io.hpp"
#include <array>

namespace gcv
{

    using Histogram = std::array<int, 256>;

    // Computes a 256-bin intensity histogram. If `img` is RGB it's converted to
    // grayscale first (via grayscale_cpu); if already single-channel it's used
    // directly.
    Histogram histogram_cpu(const Image &img);

} // namespace gcv