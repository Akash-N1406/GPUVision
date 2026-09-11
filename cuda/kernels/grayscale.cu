// cuda/kernels/grayscale.cu

#include "grayscale.cuh"
#include "../utils/cuda_utils.cuh"
#include <stdexcept>

namespace gcv {

namespace {

constexpr int THREADS_PER_BLOCK = 256;

// One thread computes one output pixel — the simplest possible mapping,
// per SRS section 8 ("Thread 0 -> Pixel 0, Thread 1 -> Pixel 1, ..."). No
// shared memory, no tiling; that optimization story starts in Phase 4/5
// with convolution, not here. Grayscale/inversion exist specifically to
// demonstrate the basic thread/block/grid model before anything fancier.
__global__ void grayscale_kernel(const uint8_t* rgb, uint8_t* gray,
                                  int num_pixels) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pixels) return; // guard: grid is rounded up to a block multiple

    const uint8_t r = rgb[idx * 3 + 0];
    const uint8_t g = rgb[idx * 3 + 1];
    const uint8_t b = rgb[idx * 3 + 2];

    // Same weights and rounding as grayscale_cpu, so CPU/GPU outputs should
    // match exactly (or within 1 due to float rounding) per FR-17.
    const float value = 0.299f * r + 0.587f * g + 0.114f * b;
    gray[idx] = static_cast<uint8_t>(value + 0.5f);
}

} // namespace

Image grayscale_cuda(const Image& rgb) {
    if (rgb.channels != 3) {
        throw std::runtime_error(
            "grayscale_cuda: expected a 3-channel RGB image, got " +
            std::to_string(rgb.channels) + " channels");
    }

    const int num_pixels = rgb.width * rgb.height;
    const size_t rgb_bytes = static_cast<size_t>(num_pixels) * 3;
    const size_t gray_bytes = static_cast<size_t>(num_pixels);

    uint8_t* d_rgb = nullptr;
    uint8_t* d_gray = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rgb, rgb_bytes));
    CUDA_CHECK(cudaMalloc(&d_gray, gray_bytes));

    CUDA_CHECK(cudaMemcpy(d_rgb, rgb.data.get(), rgb_bytes, cudaMemcpyHostToDevice));

    const int blocks = (num_pixels + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    grayscale_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_rgb, d_gray, num_pixels);
    CUDA_CHECK_KERNEL_LAUNCH();
    CUDA_CHECK(cudaDeviceSynchronize());

    Image out = make_image(rgb.width, rgb.height, 1);
    CUDA_CHECK(cudaMemcpy(out.data.get(), d_gray, gray_bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_rgb));
    CUDA_CHECK(cudaFree(d_gray));

    return out;
}

} // namespace gcv
