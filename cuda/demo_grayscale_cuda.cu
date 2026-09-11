// cuda/demo_grayscale_cuda.cu
//
// First CPU-vs-GPU comparison in the project. Detailed H2D/kernel/D2H
// timing breakdown (FR-12) comes later with the real benchmark harness in
// Phase 6 — for now this measures simple end-to-end wall time for each
// side and, more importantly, runs the FR-17 correctness check: does the
// GPU output actually match the CPU baseline?
//
// Build (from project root, on the machine with the GPU):
//   nvcc -std=c++17 -O2 -arch=sm_86 \
//       cuda/demo_grayscale_cuda.cu cuda/kernels/grayscale.cu \
//       cpp/cpu/grayscale.cpp cpp/common/image_io.cpp \
//       -I. $(pkg-config --cflags --libs opencv4) \
//       -o build_grayscale_cuda_demo
//
// (sm_86 targets your RTX 3050 Laptop's Ampere compute capability 8.6 —
//  see CMakeLists.txt for the multi-arch list used by the full build.)
//
// Run:
//   ./build_grayscale_cuda_demo data/input/<your_image>.jpg data/output/grayscale_gpu.png

#include "cpp/cpu/grayscale.hpp"
#include "cuda/kernels/grayscale.cuh"
#include <chrono>
#include <cmath>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_image> <output_image>\n";
        return 1;
    }

    const std::string input_path = argv[1];
    const std::string output_path = argv[2];

    try {
        gcv::Image rgb = gcv::load_image_rgb(input_path);
        std::cout << "Loaded " << input_path << " (" << rgb.width << "x"
                  << rgb.height << ", " << rgb.channels << " channels)\n\n";

        // --- CPU baseline ---
        const auto cpu_start = std::chrono::high_resolution_clock::now();
        gcv::Image cpu_result = gcv::grayscale_cpu(rgb);
        const auto cpu_end = std::chrono::high_resolution_clock::now();
        const double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

        // --- GPU (first call includes CUDA context init overhead, so we
        //     warm up once and then time a second call for a fairer number) ---
        gcv::grayscale_cuda(rgb); // warm-up, result discarded

        const auto gpu_start = std::chrono::high_resolution_clock::now();
        gcv::Image gpu_result = gcv::grayscale_cuda(rgb);
        const auto gpu_end = std::chrono::high_resolution_clock::now();
        const double gpu_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

        std::cout << "CPU:  " << cpu_ms << " ms\n";
        std::cout << "GPU:  " << gpu_ms << " ms (end-to-end incl. H2D/D2H transfer; "
                  << "warmed up)\n";
        std::cout << "Speedup: " << (cpu_ms / gpu_ms) << "x\n\n";

        // --- FR-17 correctness verification ---
        if (cpu_result.width != gpu_result.width || cpu_result.height != gpu_result.height ||
            cpu_result.channels != gpu_result.channels) {
            std::cerr << "FAIL: CPU and GPU output dimensions differ!\n";
            return 1;
        }

        const size_t n = cpu_result.size_bytes();
        int max_abs_error = 0;
        double sum_abs_error = 0.0;
        int num_differing = 0;

        for (size_t i = 0; i < n; ++i) {
            const int diff = std::abs(static_cast<int>(cpu_result.data[i]) -
                                       static_cast<int>(gpu_result.data[i]));
            if (diff > 0) num_differing++;
            if (diff > max_abs_error) max_abs_error = diff;
            sum_abs_error += diff;
        }

        std::cout << "Correctness check (CPU vs GPU):\n";
        std::cout << "  Max absolute error:  " << max_abs_error << "\n";
        std::cout << "  Mean absolute error: " << (sum_abs_error / n) << "\n";
        std::cout << "  Differing pixels:    " << num_differing << " / " << n
                  << " (" << (100.0 * num_differing / n) << "%)\n";

        // A max error of 0 or 1 is expected (float rounding at the 0.5f
        // boundary can occasionally round differently between CPU and GPU
        // floating-point paths). Anything larger means a real bug.
        if (max_abs_error > 1) {
            std::cerr << "\nFAIL: max error exceeds the expected float-rounding "
                         "tolerance (1). Something is wrong in the kernel.\n";
            return 1;
        }
        std::cout << "\nPASS: GPU output matches CPU baseline within tolerance.\n";

        gcv::save_image(output_path, gpu_result);
        std::cout << "Saved " << output_path << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
