// cuda/kernels/convolution.cu

#include "convolution.cuh"
#include "../utils/cuda_utils.cuh"
#include <stdexcept>

namespace gcv {

namespace {

// 2D block of 16x16 threads (256 total, same as the 1D kernels) — but
// mapped over (x, y) instead of a flat pixel index. Convolution is
// inherently a 2D-neighborhood operation, and Phase 5's tiled version
// needs a 2D block shape anyway (each block cooperatively loads a tile),
// so we set that shape up here even though the naive version doesn't
// exploit it yet.
constexpr int BLOCK_DIM = 16;

// Naive: every thread re-reads its full neighborhood straight from global
// memory, with no reuse between neighboring threads even though adjacent
// output pixels share most of their input footprint. That redundant
// global-memory traffic is exactly what Phase 5's shared-memory version
// eliminates — this kernel exists to be the "before" measurement.
__global__ void convolution_kernel_naive(const uint8_t* in, uint8_t* out,
                                          int width, int height, int channels,
                                          const float* kernel, int kernel_size) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int radius = kernel_size / 2;

    for (int c = 0; c < channels; ++c) {
        float sum = 0.0f;

        for (int ky = -radius; ky <= radius; ++ky) {
            const int sy = clamp_coord(y + ky, 0, height - 1);
            for (int kx = -radius; kx <= radius; ++kx) {
                const int sx = clamp_coord(x + kx, 0, width - 1);
                const float weight = kernel[(ky + radius) * kernel_size + (kx + radius)];
                sum += weight * in[(sy * width + sx) * channels + c];
            }
        }

        out[(y * width + x) * channels + c] = clamp_to_byte(sum);
    }
}

} // namespace

Image convolution_cuda_naive(const Image& img, const std::vector<float>& kernel,
                              int kernel_size) {
    if (kernel_size % 2 == 0) {
        throw std::runtime_error("convolution_cuda_naive: kernel_size must be odd");
    }
    if (static_cast<int>(kernel.size()) != kernel_size * kernel_size) {
        throw std::runtime_error("convolution_cuda_naive: kernel size mismatch");
    }

    const size_t img_bytes = img.size_bytes();
    const size_t kernel_bytes = kernel.size() * sizeof(float);

    uint8_t* d_in = nullptr;
    uint8_t* d_out = nullptr;
    float* d_kernel = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, img_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, img_bytes));
    CUDA_CHECK(cudaMalloc(&d_kernel, kernel_bytes));

    CUDA_CHECK(cudaMemcpy(d_in, img.data.get(), img_bytes, cudaMemcpyHostToDevice));
    // Deliberately global memory here, not __constant__ — the naive version
    // is meant to be the un-optimized baseline. Constant memory for the
    // kernel weights is one of the specific Phase 5 optimizations.
    CUDA_CHECK(cudaMemcpy(d_kernel, kernel.data(), kernel_bytes, cudaMemcpyHostToDevice));

    const dim3 block(BLOCK_DIM, BLOCK_DIM);
    const dim3 grid((img.width + BLOCK_DIM - 1) / BLOCK_DIM,
                     (img.height + BLOCK_DIM - 1) / BLOCK_DIM);

    convolution_kernel_naive<<<grid, block>>>(d_in, d_out, img.width, img.height,
                                               img.channels, d_kernel, kernel_size);
    CUDA_CHECK_KERNEL_LAUNCH();
    CUDA_CHECK(cudaDeviceSynchronize());

    Image out = make_image(img.width, img.height, img.channels);
    CUDA_CHECK(cudaMemcpy(out.data.get(), d_out, img_bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_kernel));

    return out;
}

} // namespace gcv
