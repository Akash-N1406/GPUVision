// cuda/demo_tiled_matrix_mul_cuda.cu
//
// The FR-10 equivalent of demo_tiled_convolution_cuda.cu: CPU baseline vs
// naive CUDA vs shared-memory tiled CUDA, all three cross-checked, with
// naive-vs-tiled being the number that actually demonstrates the
// optimization (not just CPU-vs-GPU).
//
// Build (from project root):
//   nvcc -std=c++17 -O2 -arch=sm_86 \
//       cuda/demo_tiled_matrix_mul_cuda.cu \
//       cuda/kernels/matrix_mul.cu cuda/optimized/tiled_matrix_mul.cu \
//       cpp/cpu/matrix_mul.cpp \
//       -I. -o build_tiled_matmul_demo
//
// Run:
//   ./build_tiled_matmul_demo [N]   (default N=512)

#include "cpp/cpu/matrix_mul.hpp"
#include "cuda/kernels/matrix_mul.cuh"
#include "cuda/optimized/tiled_matrix_mul.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>

namespace {

double max_abs_diff(const gcv::Matrix& a, const gcv::Matrix& b) {
    double max_diff = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        max_diff = std::max(max_diff, static_cast<double>(std::abs(a.data[i] - b.data[i])));
    }
    return max_diff;
}

} // namespace

int main(int argc, char** argv) {
    const int N = argc > 1 ? std::atoi(argv[1]) : 512;
    std::cout << "Benchmarking " << N << "x" << N << " x " << N << "x" << N
              << " multiplication...\n\n";

    gcv::Matrix A = gcv::make_matrix(N, N);
    gcv::Matrix B = gcv::make_matrix(N, N);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < A.size(); ++i) A.data[i] = dist(rng);
    for (size_t i = 0; i < B.size(); ++i) B.data[i] = dist(rng);

    const auto cpu_start = std::chrono::high_resolution_clock::now();
    gcv::Matrix cpu_result = gcv::matmul_cpu(A, B);
    const auto cpu_end = std::chrono::high_resolution_clock::now();
    const double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    gcv::matmul_cuda_naive(A, B); // warm-up
    const auto naive_start = std::chrono::high_resolution_clock::now();
    gcv::Matrix naive_result = gcv::matmul_cuda_naive(A, B);
    const auto naive_end = std::chrono::high_resolution_clock::now();
    const double naive_ms = std::chrono::duration<double, std::milli>(naive_end - naive_start).count();

    gcv::matmul_cuda_tiled(A, B); // warm-up
    const auto tiled_start = std::chrono::high_resolution_clock::now();
    gcv::Matrix tiled_result = gcv::matmul_cuda_tiled(A, B);
    const auto tiled_end = std::chrono::high_resolution_clock::now();
    const double tiled_ms = std::chrono::duration<double, std::milli>(tiled_end - tiled_start).count();

    std::cout << "CPU:         " << cpu_ms << " ms\n";
    std::cout << "Naive CUDA:  " << naive_ms << " ms  (speedup vs CPU: "
              << (cpu_ms / naive_ms) << "x)\n";
    std::cout << "Tiled CUDA:  " << tiled_ms << " ms  (speedup vs CPU: "
              << (cpu_ms / tiled_ms) << "x)\n";
    std::cout << "Tiled vs Naive speedup: " << (naive_ms / tiled_ms) << "x\n\n";

    const double naive_err = max_abs_diff(cpu_result, naive_result);
    const double tiled_err = max_abs_diff(cpu_result, tiled_result);
    const double cross_err = max_abs_diff(naive_result, tiled_result);

    std::cout << "Correctness (max absolute error):\n";
    std::cout << "  CPU vs Naive CUDA: " << naive_err << "\n";
    std::cout << "  CPU vs Tiled CUDA: " << tiled_err << "\n";
    std::cout << "  Naive vs Tiled:    " << cross_err << "\n";

    constexpr double TOLERANCE = 0.05;
    if (naive_err > TOLERANCE || tiled_err > TOLERANCE || cross_err > TOLERANCE) {
        std::cerr << "\nFAIL: at least one comparison exceeds tolerance ("
                  << TOLERANCE << "). If naive passes but tiled fails, check the "
                  << "shared-memory tile loading and __syncthreads placement.\n";
        return 1;
    }
    std::cout << "\nPASS: CPU, naive CUDA, and tiled CUDA all agree within tolerance.\n";

    return 0;
}
