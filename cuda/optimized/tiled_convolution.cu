// cuda/optimized/tiled_convolution.cu

#include "tiled_convolution.cuh"
#include "../utils/cuda_utils.cuh"
#include <stdexcept>

namespace gcv {

namespace {

// Each block computes a TILE_DIM x TILE_DIM tile of *output* pixels. To do
// that it needs a (TILE_DIM + 2*radius) x (TILE_DIM + 2*radius) region of
// *input* pixels in shared memory (the extra border is the "halo" — the
// neighboring pixels needed by threads at the tile's edges).
constexpr int TILE_DIM = 16;

// Loads ALL channels into shared memory in a single pass and computes all
// channels' outputs in a single kernel launch — unlike an earlier version
// of this file, which launched once per channel. That per-channel design
// measured (via cuda/demo_convolution_detailed.cu) as 1.4x-1.9x SLOWER
// than the naive global-memory kernel, and the gap grew with resolution:
// each channel was a full independent sweep over the image with zero
// cache reuse between channels, effectively running 3 full passes instead
// of 1, plus 3x the launch/__syncthreads() overhead. This version loads
// the interleaved tile (all channels together, matching the image's actual
// memory layout) once, syncs once, and every thread loops over channels
// while reading from shared memory — same channel-loop structure as the
// naive kernel, just with shared-memory reuse now applied once instead of
// redundantly 3 times.
__global__ void convolution_tiled_kernel(const uint8_t* in, uint8_t* out,
                                          int width, int height, int channels,
                                          const float* kernel, int kernel_size,
                                          int radius) {
    extern __shared__ uint8_t tile[];
    const int tile_dim = TILE_DIM + 2 * radius;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int out_x = blockIdx.x * TILE_DIM + tx;
    const int out_y = blockIdx.y * TILE_DIM + ty;

    // Cooperative load: tile is bigger than the block, so each thread may
    // load more than one tile position, strided by the block size. Every
    // position loads all `channels` bytes together (interleaved, matching
    // global memory's layout) instead of one channel at a time.
    for (int ly = ty; ly < tile_dim; ly += TILE_DIM) {
        int gy = blockIdx.y * TILE_DIM + ly - radius;
        gy = clamp_coord(gy, 0, height - 1);
        for (int lx = tx; lx < tile_dim; lx += TILE_DIM) {
            int gx = blockIdx.x * TILE_DIM + lx - radius;
            gx = clamp_coord(gx, 0, width - 1);
            for (int c = 0; c < channels; ++c) {
                tile[(ly * tile_dim + lx) * channels + c] = in[(gy * width + gx) * channels + c];
            }
        }
    }

    // ONE sync for the whole block, covering all channels — not one sync
    // per channel like the previous per-channel-launch design.
    __syncthreads();

    if (out_x < width && out_y < height) {
        for (int c = 0; c < channels; ++c) {
            float sum = 0.0f;
            for (int ky = 0; ky < kernel_size; ++ky) {
                for (int kx = 0; kx < kernel_size; ++kx) {
                    const float weight = kernel[ky * kernel_size + kx];
                    sum += weight * tile[((ty + ky) * tile_dim + (tx + kx)) * channels + c];
                }
            }
            out[(out_y * width + out_x) * channels + c] = clamp_to_byte(sum);
        }
    }
}

} // namespace

Image convolution_cuda_tiled(const Image& img, const std::vector<float>& kernel,
                              int kernel_size) {
    if (kernel_size % 2 == 0) {
        throw std::runtime_error("convolution_cuda_tiled: kernel_size must be odd");
    }
    if (static_cast<int>(kernel.size()) != kernel_size * kernel_size) {
        throw std::runtime_error("convolution_cuda_tiled: kernel size mismatch");
    }

    const int radius = kernel_size / 2;
    const size_t img_bytes = img.size_bytes();
    const size_t kernel_bytes = kernel.size() * sizeof(float);

    uint8_t* d_in = nullptr;
    uint8_t* d_out = nullptr;
    float* d_kernel = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, img_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, img_bytes));
    CUDA_CHECK(cudaMalloc(&d_kernel, kernel_bytes));

    CUDA_CHECK(cudaMemcpy(d_in, img.data.get(), img_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, kernel.data(), kernel_bytes, cudaMemcpyHostToDevice));

    const dim3 block(TILE_DIM, TILE_DIM);
    const dim3 grid((img.width + TILE_DIM - 1) / TILE_DIM,
                     (img.height + TILE_DIM - 1) / TILE_DIM);

    const int tile_dim = TILE_DIM + 2 * radius;
    const size_t shared_bytes =
        static_cast<size_t>(tile_dim) * tile_dim * img.channels * sizeof(uint8_t);

    convolution_tiled_kernel<<<grid, block, shared_bytes>>>(
        d_in, d_out, img.width, img.height, img.channels, d_kernel, kernel_size, radius);
    CUDA_CHECK_KERNEL_LAUNCH();
    CUDA_CHECK(cudaDeviceSynchronize());

    Image out = make_image(img.width, img.height, img.channels);
    CUDA_CHECK(cudaMemcpy(out.data.get(), d_out, img_bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_kernel));

    return out;
}

DetailedTimingSamples convolution_cuda_tiled_detailed(const Image& img,
                                                       const std::vector<float>& kernel,
                                                       int kernel_size,
                                                       int warmup, int measured) {
    if (kernel_size % 2 == 0) {
        throw std::runtime_error("convolution_cuda_tiled_detailed: kernel_size must be odd");
    }
    if (static_cast<int>(kernel.size()) != kernel_size * kernel_size) {
        throw std::runtime_error("convolution_cuda_tiled_detailed: kernel size mismatch");
    }

    const int radius = kernel_size / 2;
    const size_t img_bytes = img.size_bytes();
    const size_t kernel_bytes = kernel.size() * sizeof(float);

    uint8_t* d_in = nullptr;
    uint8_t* d_out = nullptr;
    float* d_kernel = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, img_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, img_bytes));
    CUDA_CHECK(cudaMalloc(&d_kernel, kernel_bytes));
    CUDA_CHECK(cudaMemcpy(d_kernel, kernel.data(), kernel_bytes, cudaMemcpyHostToDevice));

    std::vector<uint8_t> host_out(img_bytes);

    cudaEvent_t ev_start, ev_h2d, ev_kernel, ev_d2h;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_h2d));
    CUDA_CHECK(cudaEventCreate(&ev_kernel));
    CUDA_CHECK(cudaEventCreate(&ev_d2h));

    const dim3 block(TILE_DIM, TILE_DIM);
    const dim3 grid((img.width + TILE_DIM - 1) / TILE_DIM,
                     (img.height + TILE_DIM - 1) / TILE_DIM);
    const int tile_dim = TILE_DIM + 2 * radius;
    const size_t shared_bytes =
        static_cast<size_t>(tile_dim) * tile_dim * img.channels * sizeof(uint8_t);

    // Single kernel launch now, not a per-channel loop — "kernel" phase
    // measures exactly one convolution_tiled_kernel invocation.
    auto run_once = [&]() {
        CUDA_CHECK(cudaEventRecord(ev_start));
        CUDA_CHECK(cudaMemcpy(d_in, img.data.get(), img_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(ev_h2d));
        convolution_tiled_kernel<<<grid, block, shared_bytes>>>(
            d_in, d_out, img.width, img.height, img.channels, d_kernel, kernel_size, radius);
        CUDA_CHECK_KERNEL_LAUNCH();
        CUDA_CHECK(cudaEventRecord(ev_kernel));
        CUDA_CHECK(cudaMemcpy(host_out.data(), d_out, img_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(ev_d2h));
        CUDA_CHECK(cudaEventSynchronize(ev_d2h));
    };

    for (int i = 0; i < warmup; ++i) {
        run_once();
    }

    DetailedTimingSamples result;
    result.h2d_ms.reserve(measured);
    result.kernel_ms.reserve(measured);
    result.d2h_ms.reserve(measured);

    for (int i = 0; i < measured; ++i) {
        run_once();
        float h2d = 0.0f, kern = 0.0f, d2h = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&h2d, ev_start, ev_h2d));
        CUDA_CHECK(cudaEventElapsedTime(&kern, ev_h2d, ev_kernel));
        CUDA_CHECK(cudaEventElapsedTime(&d2h, ev_kernel, ev_d2h));
        result.h2d_ms.push_back(h2d);
        result.kernel_ms.push_back(kern);
        result.d2h_ms.push_back(d2h);
    }

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_h2d));
    CUDA_CHECK(cudaEventDestroy(ev_kernel));
    CUDA_CHECK(cudaEventDestroy(ev_d2h));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_kernel));

    return result;
}

} // namespace gcv
