// cuda/kernels/gaussian_blur.cu

#include "gaussian_blur.cuh"
#include "../optimized/tiled_convolution.cuh"
#include <cmath>
#include <stdexcept>
#include <vector>

namespace gcv {

namespace {

// Identical math to the private helper in cpp/cpu/gaussian_blur.cpp —
// duplicated rather than shared across a header, since it's small and
// self-contained, and keeps this file's only dependency on Phase 4's
// tested convolution engine rather than reaching back into CPU code.
std::vector<float> make_gaussian_kernel(int kernel_size, float sigma) {
    if (kernel_size % 2 == 0) {
        throw std::runtime_error("make_gaussian_kernel: kernel_size must be odd");
    }

    std::vector<float> kernel(kernel_size * kernel_size);
    const int radius = kernel_size / 2;
    float sum = 0.0f;

    for (int y = -radius; y <= radius; ++y) {
        for (int x = -radius; x <= radius; ++x) {
            const float value = std::exp(-(x * x + y * y) / (2.0f * sigma * sigma));
            kernel[(y + radius) * kernel_size + (x + radius)] = value;
            sum += value;
        }
    }
    for (float& v : kernel) v /= sum;

    return kernel;
}

} // namespace

Image gaussian_blur_cuda(const Image& img, int kernel_size, float sigma) {
    const std::vector<float> kernel = make_gaussian_kernel(kernel_size, sigma);
    return convolution_cuda_tiled(img, kernel, kernel_size);
}

} // namespace gcv
