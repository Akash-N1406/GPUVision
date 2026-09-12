// cuda/kernels/gaussian_blur.cuh
//
// FR-05 CUDA version. Gaussian blur is convolution with a specific
// generated kernel, and we already have a verified, benchmarked shared-
// memory convolution engine from Phase 4 — so this is deliberately a thin
// wrapper, not a new kernel. Keeping it as its own named file matches the
// SRS's per-feature structure and gives it its own entry point for the
// Phase 6 benchmark suite, without duplicating tested kernel code.

#pragma once

#include "../../cpp/common/image_io.hpp"

namespace gcv {

// Applies Gaussian blur on the GPU using the shared-memory tiled
// convolution kernel underneath. Same signature/defaults as
// gaussian_blur_cpu for direct comparison.
Image gaussian_blur_cuda(const Image& img, int kernel_size = 5, float sigma = 1.4f);

} // namespace gcv
