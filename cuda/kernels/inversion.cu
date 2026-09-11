// cuda/kernels/inversion.cu

#include "inversion.cuh"
#include "../utils/cuda_utils.cuh"

namespace gcv {

namespace {

constexpr int THREADS_PER_BLOCK = 256;

// 1 thread = 1 byte. No pixel/channel structure needed at all — the
// simplest possible CUDA kernel in the project, intentionally, so it's a
// clean second data point for "thread/block/grid" fundamentals right after
// grayscale.
__global__ void inversion_kernel(const uint8_t* in, uint8_t* out, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = 255 - in[idx];
}

} // namespace

Image invert_cuda(const Image& img) {
    const size_t n = img.size_bytes();

    uint8_t* d_in = nullptr;
    uint8_t* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, n));
    CUDA_CHECK(cudaMalloc(&d_out, n));

    CUDA_CHECK(cudaMemcpy(d_in, img.data.get(), n, cudaMemcpyHostToDevice));

    const int blocks = (static_cast<int>(n) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    inversion_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_in, d_out, static_cast<int>(n));
    CUDA_CHECK_KERNEL_LAUNCH();
    CUDA_CHECK(cudaDeviceSynchronize());

    Image out = make_image(img.width, img.height, img.channels);
    CUDA_CHECK(cudaMemcpy(out.data.get(), d_out, n, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    return out;
}

} // namespace gcv
