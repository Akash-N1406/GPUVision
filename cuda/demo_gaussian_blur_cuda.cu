// cuda/demo_gaussian_blur_cuda.cu
//
// Build (from project root):
//   nvcc -std=c++17 -O2 -arch=sm_86 \
//       cuda/demo_gaussian_blur_cuda.cu cuda/kernels/gaussian_blur.cu \
//       cuda/optimized/tiled_convolution.cu \
//       cpp/cpu/gaussian_blur.cpp cpp/common/convolution_utils.cpp cpp/common/image_io.cpp \
//       -I. $(pkg-config --cflags --libs opencv4) \
//       -o build_gaussian_blur_cuda_demo
//
// Run:
//   ./build_gaussian_blur_cuda_demo data/input/<your_image>.jpg data/output/blur_gpu.png [kernel_size] [sigma]

#include "cpp/cpu/gaussian_blur.hpp"
#include "cuda/kernels/gaussian_blur.cuh"
#include <chrono>
#include <cmath>
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 3 || argc > 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <input_image> <output_image> [kernel_size=5] [sigma=1.4]\n";
        return 1;
    }

    const std::string input_path = argv[1];
    const std::string output_path = argv[2];
    const int kernel_size = argc > 3 ? std::stoi(argv[3]) : 5;
    const float sigma = argc > 4 ? std::stof(argv[4]) : 1.4f;

    try {
        gcv::Image rgb = gcv::load_image_rgb(input_path);
        std::cout << "Loaded " << input_path << " (" << rgb.width << "x"
                  << rgb.height << ", " << rgb.channels << " channels)\n";
        std::cout << "Kernel: " << kernel_size << "x" << kernel_size
                  << ", sigma=" << sigma << "\n\n";

        const auto cpu_start = std::chrono::high_resolution_clock::now();
        gcv::Image cpu_result = gcv::gaussian_blur_cpu(rgb, kernel_size, sigma);
        const auto cpu_end = std::chrono::high_resolution_clock::now();
        const double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

        gcv::gaussian_blur_cuda(rgb, kernel_size, sigma); // warm-up
        const auto gpu_start = std::chrono::high_resolution_clock::now();
        gcv::Image gpu_result = gcv::gaussian_blur_cuda(rgb, kernel_size, sigma);
        const auto gpu_end = std::chrono::high_resolution_clock::now();
        const double gpu_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

        std::cout << "CPU: " << cpu_ms << " ms\n";
        std::cout << "GPU: " << gpu_ms << " ms (end-to-end incl. H2D/D2H transfer; warmed up)\n";
        std::cout << "Speedup: " << (cpu_ms / gpu_ms) << "x\n\n";

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
