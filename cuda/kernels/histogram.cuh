// cuda/kernels/histogram.cuh
//
// FR-11 CUDA version. This is the project's "memory contention" feature:
// many threads incrementing the same 256 bins concurrently needs atomic
// operations, not just independent per-thread writes like every kernel so
// far. Uses the standard privatized-histogram pattern: each block builds a
// private histogram in shared memory (block-local atomics, cheap and
// low-contention since only threads in the same block can collide), then
// merges into the global histogram with one atomicAdd per non-empty bin
// per block — far fewer, and far less contended, global atomics than if
// every pixel hit global memory directly.

#pragma once

#include "../../cpp/common/image_io.hpp"
#include <array>

namespace gcv {

using Histogram = std::array<int, 256>;

// Computes a 256-bin intensity histogram on the GPU. Converts to grayscale
// first if given an RGB image, same convention as histogram_cpu.
Histogram histogram_cuda(const Image& img);

} // namespace gcv
