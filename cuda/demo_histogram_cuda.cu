// cuda/demo_histogram_cuda.cu
//
// Build (from project root):
//   nvcc -std=c++17 -O2 -arch=sm_86 \
//       cuda/demo_histogram_cuda.cu cuda/kernels/histogram.cu cuda/kernels/grayscale.cu \
//       cpp/cpu/histogram.cpp cpp/cpu/grayscale.cpp cpp/common/image_io.cpp \
//       -I. $(pkg-config --cflags --libs opencv4) \
//       -o build_histogram_cuda_demo
//
// Run:
//   ./build_histogram_cuda_demo data/input/<your_image>.jpg

#include "cpp/cpu/histogram.hpp"
#include "cuda/kernels/histogram.cuh"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_image>\n";
        return 1;
    }

    const std::string input_path = argv[1];

    try {
        gcv::Image rgb = gcv::load_image_rgb(input_path);
        std::cout << "Loaded " << input_path << " (" << rgb.width << "x"
                  << rgb.height << ", " << rgb.channels << " channels)\n\n";

        const auto cpu_start = std::chrono::high_resolution_clock::now();
        gcv::Histogram cpu_hist = gcv::histogram_cpu(rgb);
        const auto cpu_end = std::chrono::high_resolution_clock::now();
        const double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

        gcv::histogram_cuda(rgb); // warm-up
        const auto gpu_start = std::chrono::high_resolution_clock::now();
        gcv::Histogram gpu_hist = gcv::histogram_cuda(rgb);
        const auto gpu_end = std::chrono::high_resolution_clock::now();
        const double gpu_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

        std::cout << "CPU: " << cpu_ms << " ms\n";
        std::cout << "GPU: " << gpu_ms << " ms (end-to-end incl. H2D/D2H transfer; warmed up)\n";
        std::cout << "Speedup: " << (cpu_ms / gpu_ms) << "x\n\n";

        // Histogram counting is exact integer arithmetic — unlike every
        // previous kernel, there's no floating-point rounding anywhere in
        // this one. If atomics are working correctly, every single bin
        // must match exactly. Any mismatch here means a real race
        // condition (a lost update), not noise.
        bool exact_match = true;
        int max_bin_diff = 0;
        for (int i = 0; i < 256; ++i) {
            const int diff = std::abs(cpu_hist[i] - gpu_hist[i]);
            if (diff > 0) exact_match = false;
            max_bin_diff = std::max(max_bin_diff, diff);
        }

        long long cpu_total = 0, gpu_total = 0;
        for (int i = 0; i < 256; ++i) {
            cpu_total += cpu_hist[i];
            gpu_total += gpu_hist[i];
        }

        std::cout << "Correctness check (CPU vs GPU):\n";
        std::cout << "  CPU total count: " << cpu_total << "\n";
        std::cout << "  GPU total count: " << gpu_total << "\n";
        std::cout << "  Max bin difference: " << max_bin_diff << "\n";
        std::cout << "  Exact match: " << (exact_match ? "YES" : "NO") << "\n";

        if (!exact_match) {
            std::cerr << "\nFAIL: histogram bins don't match exactly. This means "
                         "atomicAdd usage has a real bug (e.g. missing __syncthreads, "
                         "or a race in the merge step) — not float rounding.\n";
            return 1;
        }
        std::cout << "\nPASS: GPU histogram matches CPU baseline exactly.\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
