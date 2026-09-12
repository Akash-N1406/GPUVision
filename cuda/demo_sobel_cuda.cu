// cuda/demo_sobel_cuda.cu
//
// Build (from project root):
//   nvcc -std=c++17 -O2 -arch=sm_86 \
//       cuda/demo_sobel_cuda.cu cuda/kernels/sobel.cu cuda/kernels/grayscale.cu \
//       cpp/cpu/sobel.cpp cpp/cpu/grayscale.cpp cpp/common/convolution_utils.cpp \
//       cpp/common/image_io.cpp \
//       -I. $(pkg-config --cflags --libs opencv4) \
//       -o build_sobel_cuda_demo
//
// Run:
//   ./build_sobel_cuda_demo data/input/<your_image>.jpg data/output/sobel_gpu.png

#include "cpp/cpu/sobel.hpp"
#include "cuda/kernels/sobel.cuh"
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

        const auto cpu_start = std::chrono::high_resolution_clock::now();
        gcv::Image cpu_result = gcv::sobel_cpu(rgb);
        const auto cpu_end = std::chrono::high_resolution_clock::now();
        const double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

        gcv::sobel_cuda(rgb); // warm-up
        const auto gpu_start = std::chrono::high_resolution_clock::now();
        gcv::Image gpu_result = gcv::sobel_cuda(rgb);
        const auto gpu_end = std::chrono::high_resolution_clock::now();
        const double gpu_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

        std::cout << "CPU: " << cpu_ms << " ms\n";
        std::cout << "GPU: " << gpu_ms << " ms (end-to-end incl. H2D/D2H transfer; warmed up)\n";
        std::cout << "Speedup: " << (cpu_ms / gpu_ms) << "x\n\n";

        if (cpu_result.width != gpu_result.width || cpu_result.height != gpu_result.height) {
            std::cerr << "FAIL: CPU and GPU output dimensions differ!\n";
            return 1;
        }

        const size_t n = cpu_result.size_bytes();
        int max_abs_error = 0;
        int num_differing = 0;
        for (size_t i = 0; i < n; ++i) {
            const int diff = std::abs(static_cast<int>(cpu_result.data[i]) -
                                       static_cast<int>(gpu_result.data[i]));
            if (diff > 0) num_differing++;
            if (diff > max_abs_error) max_abs_error = diff;
        }

        std::cout << "Correctness check (CPU vs GPU):\n";
        std::cout << "  Max absolute error: " << max_abs_error << "\n";
        std::cout << "  Differing bytes:    " << num_differing << " / " << n << "\n";

        // sqrtf introduces a bit more floating-point variance than the
        // pure multiply-accumulate in blur/convolution, so allow a
        // slightly wider tolerance.
        if (max_abs_error > 2) {
            std::cerr << "\nFAIL: max error exceeds tolerance (2).\n";
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
