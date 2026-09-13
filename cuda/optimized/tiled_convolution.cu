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

// One channel at a time (host loops over channels) — this keeps shared
// memory usage small and simple (a single uint8_t per pixel in the tile,
// not one per channel), at the cost of one kernel launch per channel
// instead of one launch total. For a 3-channel image that's 3 launches
// instead of 1; at typical image sizes the extra launch overhead is
// negligible next to the memory-traffic savings shared memory buys us.
__global__ void convolution_tiled_channel_kernel(const uint8_t* in, uint8_t* out,
                                                   int width, int height, int channels,
                                                   int channel_idx, const float* kernel,
                                                   int kernel_size, int radius) {
    extern __shared__ uint8_t tile[];
    const int tile_dim = TILE_DIM + 2 * radius;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int out_x = blockIdx.x * TILE_DIM + tx;
    const int out_y = blockIdx.y * TILE_DIM + ty;

    // Cooperative load: the tile is bigger than the block (TILE_DIM x
    // TILE_DIM threads, but tile_dim x tile_dim elements needed), so each
    // thread may load more than one element, strided by the block size.
    for (int ly = ty; ly < tile_dim; ly += TILE_DIM) {
        int gy = blockIdx.y * TILE_DIM + ly - radius;
        gy = clamp_coord(gy, 0, height - 1);
        for (int lx = tx; lx < tile_dim; lx += TILE_DIM) {
            int gx = blockIdx.x * TILE_DIM + lx - radius;
            gx = clamp_coord(gx, 0, width - 1);
            tile[ly * tile_dim + lx] = in[(gy * width + gx) * channels + channel_idx];
        }
    }

    // Every thread in the block must finish loading before any thread
    // starts reading — otherwise a fast thread could read a shared-memory
    // slot a slower thread hasn't written yet.
    __syncthreads();

    if (out_x < width && out_y < height) {
        float sum = 0.0f;
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                const float weight = kernel[ky * kernel_size + kx];
                sum += weight * tile[(ty + ky) * tile_dim + (tx + kx)];
            }
        }
        out[(out_y * width + out_x) * channels + channel_idx] = clamp_to_byte(sum);
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
    const size_t shared_bytes = static_cast<size_t>(tile_dim) * tile_dim * sizeof(uint8_t);

    for (int c = 0; c < img.channels; ++c) {
        convolution_tiled_channel_kernel<<<grid, block, shared_bytes>>>(
            d_in, d_out, img.width, img.height, img.channels, c, d_kernel,
            kernel_size, radius);
        CUDA_CHECK_KERNEL_LAUNCH();
    }
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
    const size_t shared_bytes = static_cast<size_t>(tile_dim) * tile_dim * sizeof(uint8_t);

    // "kernel" phase spans ALL per-channel launches (3 for RGB) — they're
    // issued back-to-back on the default stream, so the elapsed time
    // between ev_h2d and ev_kernel naturally sums all of them, including
    // any inter-launch dispatch gaps. This is exactly the number needed to
    // test whether 3 launches' overhead outweighs shared memory's savings.
    auto run_once = [&]() {
        CUDA_CHECK(cudaEventRecord(ev_start));
        CUDA_CHECK(cudaMemcpy(d_in, img.data.get(), img_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(ev_h2d));
        for (int c = 0; c < img.channels; ++c) {
            convolution_tiled_channel_kernel<<<grid, block, shared_bytes>>>(
                d_in, d_out, img.width, img.height, img.channels, c, d_kernel,
                kernel_size, radius);
            CUDA_CHECK_KERNEL_LAUNCH();
        }
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
