// cuda/optimized/tiled_convolution.cuh
//
// FR-08/FR-09: shared-memory + tiled convolution, per SRS sections 13-14.
// This is the "after" measurement to convolution.cu's naive "before" —
// same math, same clamp-to-edge borders, same output, but each block
// cooperatively loads its input tile (plus a halo/apron for the kernel
// radius) into shared memory ONCE, then every thread reads its
// neighborhood from shared memory instead of re-hitting global memory for
// every overlapping pixel. This is the single most important comparison
// in the whole project for demonstrating real CUDA optimization skill.

#pragma once

#include "../../cpp/common/image_io.hpp"
#include <vector>

namespace gcv {

// Same signature and behavior as convolution_cuda_naive — a drop-in
// replacement, so benchmark code can call either one and diff the outputs
// (should match exactly) and the timings (should show shared-memory's win).
Image convolution_cuda_tiled(const Image& img, const std::vector<float>& kernel,
                              int kernel_size);

} // namespace gcv
