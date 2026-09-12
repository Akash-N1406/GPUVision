// cuda/kernels/convolution.cuh
//
// FR-07 CUDA version, naive global-memory implementation (SRS section 12).
// Each thread computes one output pixel, reading its neighborhood directly
// from global memory on every access — no shared memory, no tiling. This
// is intentional: it's the baseline the Phase 5 shared-memory/tiled version
// gets benchmarked against, so the whole point is that it's NOT optimized
// yet (same role grayscale/inversion played for basic thread/block/grid).

#pragma once

#include "../../cpp/common/image_io.hpp"
#include <vector>

namespace gcv {

// Applies an arbitrary odd-sized kernel to `img` on the GPU, naive
// global-memory version. Mirrors convolution_cpu's signature and behavior
// (same clamp-to-edge border handling) so outputs are directly comparable.
Image convolution_cuda_naive(const Image& img, const std::vector<float>& kernel,
                              int kernel_size);

} // namespace gcv
