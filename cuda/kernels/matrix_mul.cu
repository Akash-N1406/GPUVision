// cuda/kernels/matrix_mul.cu

#include "matrix_mul.cuh"
#include "../utils/cuda_utils.cuh"
#include <stdexcept>

namespace gcv {

namespace {

constexpr int BLOCK_DIM = 16;

// One thread = one output element, per SRS section 15. Every thread
// re-reads its entire row of A and column of B straight from global
// memory with zero reuse between neighboring threads — even though thread
// (row, col) and thread (row, col+1) read the exact same row of A. That
// redundant global traffic is exactly what Phase 5's shared-memory tiled
// version eliminates, same story as naive vs tiled convolution.
__global__ void matmul_kernel_naive(const float* A, const float* B, float* C,
                                     int M, int K, int N) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

} // namespace

Matrix matmul_cuda_naive(const Matrix& A, const Matrix& B) {
    if (A.cols != B.rows) {
        throw std::runtime_error(
            "matmul_cuda_naive: dimension mismatch (A is " + std::to_string(A.rows) +
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

    const dim3 block(BLOCK_DIM, BLOCK_DIM);
    const dim3 grid((N + BLOCK_DIM - 1) / BLOCK_DIM, (M + BLOCK_DIM - 1) / BLOCK_DIM);

    matmul_kernel_naive<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
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
