// cpp/cpu/demo_matrix_mul.cpp
//
// No OpenCV/image I/O needed for this one — pure matrix math.
//
// Build (from project root):
//   g++ -std=c++17 -O2 cpp/cpu/demo_matrix_mul.cpp cpp/cpu/matrix_mul.cpp -I. \
//       -o build_matmul_demo
//
// Run:
//   ./build_matmul_demo [N]     (default N=512, multiplies two NxN matrices)

#include "matrix_mul.hpp"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>

namespace
{

    // Fills A and B with a small, hand-verifiable case:
    //   A = [[1, 2], [3, 4]]
    //   B = [[5, 6], [7, 8]]
    //   Expected C = [[19, 22], [43, 50]]
    bool run_correctness_check()
    {
        gcv::Matrix A = gcv::make_matrix(2, 2);
        A.at(0, 0) = 1;
        A.at(0, 1) = 2;
        A.at(1, 0) = 3;
        A.at(1, 1) = 4;

        gcv::Matrix B = gcv::make_matrix(2, 2);
        B.at(0, 0) = 5;
        B.at(0, 1) = 6;
        B.at(1, 0) = 7;
        B.at(1, 1) = 8;

        gcv::Matrix C = gcv::matmul_cpu(A, B);

        const float expected[2][2] = {{19, 22}, {43, 50}};
        bool ok = true;
        for (int i = 0; i < 2 && ok; ++i)
        {
            for (int j = 0; j < 2 && ok; ++j)
            {
                if (C.at(i, j) != expected[i][j])
                    ok = false;
            }
        }

        std::cout << "Correctness check (2x2 hand-computed case): "
                  << (ok ? "PASS" : "FAIL") << "\n";
        if (!ok)
        {
            std::cout << "  Got: [[" << C.at(0, 0) << ", " << C.at(0, 1) << "], ["
                      << C.at(1, 0) << ", " << C.at(1, 1) << "]]\n";
            std::cout << "  Expected: [[19, 22], [43, 50]]\n";
        }
        return ok;
    }

} // namespace

int main(int argc, char **argv)
{
    if (!run_correctness_check())
    {
        return 1;
    }

    const int N = argc > 1 ? std::atoi(argv[1]) : 512;
    std::cout << "\nBenchmarking " << N << "x" << N << " x " << N << "x" << N
              << " multiplication...\n";

    gcv::Matrix A = gcv::make_matrix(N, N);
    gcv::Matrix B = gcv::make_matrix(N, N);

    std::mt19937 rng(42); // fixed seed for reproducibility across runs
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < A.size(); ++i)
        A.data[i] = dist(rng);
    for (size_t i = 0; i < B.size(); ++i)
        B.data[i] = dist(rng);

    const auto start = std::chrono::high_resolution_clock::now();
    gcv::Matrix C = gcv::matmul_cpu(A, B);
    const auto end = std::chrono::high_resolution_clock::now();

    const double ms = std::chrono::duration<double, std::milli>(end - start).count();

    // Checksum, not full correctness — but useful as a quick sanity value to
    // compare against once the CUDA version exists (should match closely,
    // within floating-point accumulation error).
    double checksum = 0.0;
    for (size_t i = 0; i < C.size(); ++i)
        checksum += C.data[i];

    std::cout << "CPU matmul: " << ms << " ms\n";
    std::cout << "Checksum (sum of all C elements): " << checksum << "\n";

    return 0;
}