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
#include "../utils/cuda_utils.cuh"
#include <vector>

namespace gcv {

// Applies an arbitrary odd-sized kernel to `img` on the GPU, naive
// global-memory version. Mirrors convolution_cpu's signature and behavior
// (same clamp-to-edge border handling) so outputs are directly comparable.
Image convolution_cuda_naive(const Image& img, const std::vector<float>& kernel,
                              int kernel_size);

// Instrumented version for Phase 6 benchmarking: allocates device buffers
// ONCE (not per call, unlike convolution_cuda_naive above) and reuses them
// across all warmup+measured iterations, timing H2D transfer, kernel
// execution, and D2H transfer separately via CUDA events. This avoids the
// cudaMalloc/cudaFree noise that polluted the resolution-sweep benchmark's
// wall-clock numbers, and gives the real FR-12 breakdown.
DetailedTimingSamples convolution_cuda_naive_detailed(const Image& img,
                                                       const std::vector<float>& kernel,
                                                       int kernel_size,
                                                       int warmup = 5, int measured = 20);

} // namespace gcv
