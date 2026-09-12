// cuda/kernels/histogram.cu

#include "histogram.cuh"
#include "grayscale.cuh"
#include "../utils/cuda_utils.cuh"
#include <algorithm>
#include <stdexcept>

namespace gcv {

namespace {

constexpr int THREADS_PER_BLOCK = 256;
// Capped block count with a grid-stride loop below, rather than one block
// per 256-pixel chunk. For a large image that would launch thousands of
// blocks each doing a tiny amount of work before the final merge; capping
// it keeps the per-block shared-memory histogram doing meaningful work
// before it has to sync and merge.
constexpr int MAX_BLOCKS = 256;

__global__ void histogram_kernel(const uint8_t* data, int n, int* global_hist) {
    __shared__ int local_hist[256];

    // Cooperatively zero the block's private histogram — 256 bins, however
    // many threads per block, so each thread zeroes a strided subset.
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        local_hist[i] = 0;
    }
    __syncthreads();

    // Grid-stride loop: each thread processes multiple pixels if the image
    // is larger than (blocks * threads), so block/grid size doesn't need
    // to scale with image size.
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        atomicAdd(&local_hist[data[i]], 1);
    }
    __syncthreads();

    // Merge this block's private histogram into the global one. Only 256
    // global atomics per block (not one per pixel), and each only fires
    // for bins this block actually touched.
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        if (local_hist[i] > 0) {
            atomicAdd(&global_hist[i], local_hist[i]);
        }
    }
}

} // namespace

Histogram histogram_cuda(const Image& img) {
    Image gray_owned;
    const Image* gray_ptr = &img;
    if (img.channels != 1) {
        gray_owned = grayscale_cuda(img);
        gray_ptr = &gray_owned;
    }
    const Image& gray = *gray_ptr;
    if (gray.channels != 1) {
        throw std::runtime_error("histogram_cuda: expected a 1 or 3 channel image");
    }

    const int n = static_cast<int>(gray.size_bytes());

    uint8_t* d_data = nullptr;
    int* d_hist = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, n));
    CUDA_CHECK(cudaMalloc(&d_hist, 256 * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_data, gray.data.get(), n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_hist, 0, 256 * sizeof(int)));

    const int blocks = std::min(MAX_BLOCKS, (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    histogram_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_data, n, d_hist);
    CUDA_CHECK_KERNEL_LAUNCH();
    CUDA_CHECK(cudaDeviceSynchronize());

    Histogram hist{};
    CUDA_CHECK(cudaMemcpy(hist.data(), d_hist, 256 * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_hist));

    return hist;
}

} // namespace gcv
