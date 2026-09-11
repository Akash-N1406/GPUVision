// cuda/kernels/grayscale.cuh
//
// FR-03 CUDA version: 1 thread = 1 pixel, per SRS section 8.
// This header declares the plain-C++-callable host wrapper so demo/
// benchmark code doesn't need to know any CUDA syntax — just call
// gcv::grayscale_cuda(image) like any other function.

#pragma once

#include "../../cpp/common/image_io.hpp"

namespace gcv {

// Converts an RGB Image to grayscale on the GPU. Handles all device memory
// allocation, host<->device transfer, and kernel launch internally.
// Throws std::runtime_error if rgb.channels != 3.
Image grayscale_cuda(const Image& rgb);

} // namespace gcv
