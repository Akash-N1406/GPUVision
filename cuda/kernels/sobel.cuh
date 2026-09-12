// cuda/kernels/sobel.cuh
//
// FR-06 CUDA version. Unlike generic convolution (FR-07), Sobel's Gx/Gy
// kernels are fixed, known-at-compile-time constants — not something the
// caller configures at runtime — which makes them a natural fit for CUDA
// __constant__ memory instead of the global-memory kernel-weight buffer
// convolution.cu uses. Combined with the same shared-memory tiling from
// Phase 4, this kernel demonstrates both optimizations together.

#pragma once

#include "../../cpp/common/image_io.hpp"

namespace gcv {

// Computes Sobel edge magnitude on the GPU. Converts to grayscale first
// (via grayscale_cuda) if given an RGB image. Same output convention as
// sobel_cpu: single-channel magnitude image.
Image sobel_cuda(const Image& img);

} // namespace gcv
