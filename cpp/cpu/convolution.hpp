// cpp/cpu/convolution.hpp
//
// FR-07: generic, configurable 2D convolution engine, CPU baseline.
//
// Gaussian blur and Sobel are both convolution under the hood, but the SRS
// calls out a general-purpose convolution engine as its own requirement —
// "The system will support configurable convolution kernels" — because the
// CUDA version (Phase 4) is the piece that gets the naive -> shared-memory
// -> tiled optimization treatment. This file is the CPU baseline that
// those GPU versions get diffed against, using a kernel the user picks at
// runtime rather than one hardcoded into the algorithm.

#pragma once

#include "../common/image_io.hpp"
#include <string>
#include <vector>

namespace gcv
{

    // Applies an arbitrary odd-sized kernel to `img`. This is a thin,
    // intentionally public wrapper around apply_kernel_cpu — kept as its own
    // named entry point (rather than making callers reach into
    // common/convolution_utils.hpp directly) so FR-07 has a stable API that
    // mirrors what cuda/kernels/convolution.cu will expose in Phase 4.
    Image convolution_cpu(const Image &img, const std::vector<float> &kernel,
                          int kernel_size);

    // Named preset kernels, for demo/benchmarking convenience. Each returns a
    // row-major kernel and implicitly is 3x3 (kernel_size = 3).
    std::vector<float> kernel_sharpen();      // emphasizes edges, keeps overall brightness
    std::vector<float> kernel_edge_detect();  // Laplacian-style, sum = 0 -> near-black except edges
    std::vector<float> kernel_emboss();       // directional relief effect
    std::vector<float> kernel_box_blur_3x3(); // uniform (non-Gaussian) blur, for contrast with FR-05

    // Looks up a preset by name ("sharpen", "edge", "emboss", "box_blur").
    // Throws std::runtime_error if the name isn't recognized.
    std::vector<float> get_preset_kernel(const std::string &name);

} // namespace gcv