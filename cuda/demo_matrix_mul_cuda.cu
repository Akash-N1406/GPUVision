// cuda/demo_matrix_mul_cuda.cu
//
// Build (from project root):
//   nvcc -std=c++17 -O2 -arch=sm_86 \
//       cuda/demo_matrix_mul_cuda.cu cuda/kernels/matrix_mul.cu cpp/cpu/matrix_mul.cpp \
//       -I. -o build_matmul_cuda_demo
//
// Run:
//   ./build_matmul_cuda_demo [N]   (default N=512, multiplies two NxN matrices)

#include "cpp/cpu/matrix_mul.hpp"
#include "cuda/kernels/matrix_mul.cuh"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>

int main(int argc, char** argv) {
    const int N = argc > 1 ? std::atoi(argv[1]) : 512;
    std::cout << "Benchmarking " << N << "x" << N << " x " << N << "x" << N
              << " multiplication...\n\n";

    gcv::Matrix A = gcv::make_matrix(N, N);
    gcv::Matrix B = gcv::make_matrix(N, N);

    std::mt19937 rng(42); // same seed as the Phase 2 CPU-only demo
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < A.size(); ++i) A.data[i] = dist(rng);
    for (size_t i = 0; i < B.size(); ++i) B.data[i] = dist(rng);

    const auto cpu_start = std::chrono::high_resolution_clock::now();
    gcv::Matrix cpu_result = gcv::matmul_cpu(A, B);
    const auto cpu_end = std::chrono::high_resolution_clock::now();
    const double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    gcv::matmul_cuda_naive(A, B); // warm-up
    const auto gpu_start = std::chrono::high_resolution_clock::now();
    gcv::Matrix gpu_result = gcv::matmul_cuda_naive(A, B);
    const auto gpu_end = std::chrono::high_resolution_clock::now();
    const double gpu_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

    std::cout << "CPU:  " << cpu_ms << " ms\n";
    std::cout << "GPU:  " << gpu_ms << " ms (end-to-end incl. H2D/D2H transfer; warmed up)\n";
    std::cout << "Speedup: " << (cpu_ms / gpu_ms) << "x\n\n";

    // Unlike the pixel-based kernels (single sum of ~9 terms), matmul
    // accumulates N multiply-adds per element (512 for N=512). CPU and GPU
    // may use FMA instructions in a different order, so exact bit-identical
    // results aren't guaranteed the way they were for grayscale/inversion —
    // but the difference should still be tiny relative to the value's
    // magnitude, not a sign of a real bug.
    double max_abs_error = 0.0;
    double sum_abs_error = 0.0;
    for (size_t i = 0; i < cpu_result.size(); ++i) {
        const double diff = std::abs(cpu_result.data[i] - gpu_result.data[i]);
        max_abs_error = std::max(max_abs_error, diff);
        sum_abs_error += diff;
    }
    const double mean_abs_error = sum_abs_error / cpu_result.size();

    std::cout << "Correctness check (CPU vs GPU):\n";
    std::cout << "  Max absolute error:  " << max_abs_error << "\n";
    std::cout << "  Mean absolute error: " << mean_abs_error << "\n";

    // With K up to a few thousand and values in [-1, 1], accumulated sums
    // have magnitude on the order of sqrt(K) (~20-70 for typical N here).
    // A tolerance of 0.05 is generous relative to that but still tight
    // enough to catch a real indexing/logic bug, which tends to produce
    // errors orders of magnitude larger, not a borderline miss.
    constexpr double TOLERANCE = 0.05;
    if (max_abs_error > TOLERANCE) {
        std::cerr << "\nFAIL: max error (" << max_abs_error << ") exceeds tolerance ("
                  << TOLERANCE << "). This is larger than expected float-accumulation "
                  << "drift — check the kernel's row/column indexing.\n";
        return 1;
    }
    std::cout << "\nPASS: GPU output matches CPU baseline within float-accumulation tolerance.\n";

    return 0;
}
