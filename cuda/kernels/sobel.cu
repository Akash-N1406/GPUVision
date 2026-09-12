// cuda/kernels/sobel.cu

#include "sobel.cuh"
#include "grayscale.cuh"
#include "../utils/cuda_utils.cuh"
#include <cmath>
#include <stdexcept>

namespace gcv {

namespace {

constexpr int RADIUS = 1;       // 3x3 kernels
constexpr int TILE_DIM = 16;    // output tile per block, same as tiled_convolution

// Fixed Sobel kernels (SRS section 11) live in constant memory: every
// thread in the grid reads the exact same 9 values, so the constant-memory
// cache broadcasts them in a single transaction per warp instead of each
// thread pulling from global memory — a genuinely different optimization
// from the shared-memory tiling used for the input pixels below.
__constant__ float c_sobel_kx[9];
__constant__ float c_sobel_ky[9];

__global__ void sobel_kernel(const uint8_t* gray, uint8_t* out, int width, int height) {
    extern __shared__ uint8_t tile[];
    const int tile_dim = TILE_DIM + 2 * RADIUS;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int out_x = blockIdx.x * TILE_DIM + tx;
    const int out_y = blockIdx.y * TILE_DIM + ty;

    // Cooperative tile+halo load, same pattern as tiled_convolution.cu
    // (single channel here, so no channel stride needed).
    for (int ly = ty; ly < tile_dim; ly += TILE_DIM) {
        int gy = blockIdx.y * TILE_DIM + ly - RADIUS;
        gy = clamp_coord(gy, 0, height - 1);
        for (int lx = tx; lx < tile_dim; lx += TILE_DIM) {
            int gx = blockIdx.x * TILE_DIM + lx - RADIUS;
            gx = clamp_coord(gx, 0, width - 1);
            tile[ly * tile_dim + lx] = gray[gy * width + gx];
        }
    }
    __syncthreads();

    if (out_x < width && out_y < height) {
        float gx = 0.0f;
        float gy = 0.0f;
        for (int ky = 0; ky < 3; ++ky) {
            for (int kx = 0; kx < 3; ++kx) {
                const float val = tile[(ty + ky) * tile_dim + (tx + kx)];
                gx += c_sobel_kx[ky * 3 + kx] * val;
                gy += c_sobel_ky[ky * 3 + kx] * val;
            }
        }
        const float magnitude = sqrtf(gx * gx + gy * gy);
        out[out_y * width + out_x] = clamp_to_byte(magnitude);
    }
}

} // namespace

Image sobel_cuda(const Image& img) {
    // Same move-only Image pattern as sobel_cpu: hold an optional owned
    // grayscale copy and point at whichever buffer is live.
    Image gray_owned;
    const Image* gray_ptr = &img;
    if (img.channels != 1) {
        gray_owned = grayscale_cuda(img);
        gray_ptr = &gray_owned;
    }
    const Image& gray = *gray_ptr;
    if (gray.channels != 1) {
        throw std::runtime_error("sobel_cuda: expected a 1 or 3 channel image");
    }

    static const float host_kx[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    static const float host_ky[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    CUDA_CHECK(cudaMemcpyToSymbol(c_sobel_kx, host_kx, sizeof(host_kx)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_sobel_ky, host_ky, sizeof(host_ky)));

    const size_t n = gray.size_bytes();
    uint8_t* d_gray = nullptr;
    uint8_t* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_gray, n));
    CUDA_CHECK(cudaMalloc(&d_out, n));
    CUDA_CHECK(cudaMemcpy(d_gray, gray.data.get(), n, cudaMemcpyHostToDevice));

    const dim3 block(TILE_DIM, TILE_DIM);
    const dim3 grid((gray.width + TILE_DIM - 1) / TILE_DIM,
                     (gray.height + TILE_DIM - 1) / TILE_DIM);
    const int tile_dim = TILE_DIM + 2 * RADIUS;
    const size_t shared_bytes = static_cast<size_t>(tile_dim) * tile_dim * sizeof(uint8_t);

    sobel_kernel<<<grid, block, shared_bytes>>>(d_gray, d_out, gray.width, gray.height);
    CUDA_CHECK_KERNEL_LAUNCH();
    CUDA_CHECK(cudaDeviceSynchronize());

    Image out = make_image(gray.width, gray.height, 1);
    CUDA_CHECK(cudaMemcpy(out.data.get(), d_out, n, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_gray));
    CUDA_CHECK(cudaFree(d_out));

    return out;
}

} // namespace gcv
