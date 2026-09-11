// cuda/kernels/inversion.cuh
//
// FR-04 CUDA version: 1 thread = 1 byte (per SRS section 9 — even simpler
// than grayscale since there's no channel mixing, just P_new = 255 - P_old
// applied independently to every byte).

#pragma once

#include "../../cpp/common/image_io.hpp"

namespace gcv {

// Inverts every byte of `img` on the GPU. Works for RGB or single-channel
// images alike, same as invert_cpu.
Image invert_cuda(const Image& img);

} // namespace gcv
