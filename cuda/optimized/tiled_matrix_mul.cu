// cuda/optimized/tiled_matrix_mul.cu

#include "tiled_matrix_mul.cuh"
#include "../utils/cuda_utils.cuh"
#include <stdexcept>

namespace gcv {

namespace {

constexpr int TILE_DIM = 16;

// Each block computes a TILE_DIM x TILE_DIM tile of C by sliding along the
// K dimension one TILE_DIM-wide slab at a time. For each slab: cooperatively
// load a tile of A and a tile of B into shared memory (each element loaded
// once by one thread), sync, then every thread in the block does TILE_DIM
// multiply-adds pulling from shared memory — TILE_DIM-way reuse of data
// that naive would have re-fetched from global memory for every thread.
__global__ void matmul_kernel_tiled(const float* A, const float* B, float* C,
                                     int M, int K, int N) {
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    const int row = blockIdx.y * TILE_DIM + threadIdx.y;
    const int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float sum = 0.0f;
    const int num_tiles = (K + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < num_tiles; ++t) {
        const int a_col = t * TILE_DIM + threadIdx.x;
        const int b_row = t * TILE_DIM + threadIdx.y;

        // Zero-pad out-of-bounds tile elements (needed when M, N, or K
        // isn't an exact multiple of TILE_DIM) — contributes 0 to the sum,
        // same as simply not reading it, but keeps every thread in the
        // block participating in the load so __syncthreads() below is safe.
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_DIM; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        // Must sync again before the next iteration overwrites As/Bs —
        // otherwise a fast thread could start loading the next tile while
        // a slow thread is still reading the current one.
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

} // namespace

Matrix matmul_cuda_tiled(const Matrix& A, const Matrix& B) {
    if (A.cols != B.rows) {
        throw std::runtime_error(
            "matmul_cuda_tiled: dimension mismatch (A is " + std::to_string(A.rows) +
            "x" + std::to_string(A.cols) + ", B is " + std::to_string(B.rows) +
            "x" + std::to_string(B.cols) + ")");
    }

    const int M = A.rows, K = A.cols, N = B.cols;

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, static_cast<size_t>(M) * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, static_cast<size_t>(K) * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, static_cast<size_t>(M) * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, A.data.get(), static_cast<size_t>(M) * K * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data.get(), static_cast<size_t>(K) * N * sizeof(float),
                          cudaMemcpyHostToDevice));

    const dim3 block(TILE_DIM, TILE_DIM);
    const dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    matmul_kernel_tiled<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK_KERNEL_LAUNCH();
    CUDA_CHECK(cudaDeviceSynchronize());

    Matrix C = make_matrix(M, N);
    CUDA_CHECK(cudaMemcpy(C.data.get(), d_C, static_cast<size_t>(M) * N * sizeof(float),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return C;
}

} // namespace gcv
