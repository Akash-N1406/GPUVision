// cuda/demo_convolution_cuda.cu
//
// Build (from project root):
//   nvcc -std=c++17 -O2 -arch=sm_86 \
//       cuda/demo_convolution_cuda.cu cuda/kernels/convolution.cu \
//       cpp/cpu/convolution.cpp cpp/common/convolution_utils.cpp cpp/common/image_io.cpp \
//       -I. $(pkg-config --cflags --libs opencv4) \
//       -o build_convolution_cuda_demo
//
// Run:
//   ./build_convolution_cuda_demo data/input/<your_image>.jpg data/output/conv_gpu.png sharpen
//   (preset: sharpen | edge | emboss | box_blur)

#include "cpp/cpu/convolution.hpp"
#include "cuda/kernels/convolution.cuh"
#include <chrono>
#include <cmath>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <input_image> <output_image> <preset: sharpen|edge|emboss|box_blur>\n";
        return 1;
    }

    const std::string input_path = argv[1];
    const std::string output_path = argv[2];
    const std::string preset = argv[3];

    try {
        const std::vector<float> kernel = gcv::get_preset_kernel(preset);
        constexpr int kernel_size = 3;

        gcv::Image rgb = gcv::load_image_rgb(input_path);
        std::cout << "Loaded " << input_path << " (" << rgb.width << "x"
                  << rgb.height << ", " << rgb.channels << " channels)\n";
        std::cout << "Preset: " << preset << "\n\n";

        const auto cpu_start = std::chrono::high_resolution_clock::now();
        gcv::Image cpu_result = gcv::convolution_cpu(rgb, kernel, kernel_size);
        const auto cpu_end = std::chrono::high_resolution_clock::now();
        const double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

        gcv::convolution_cuda_naive(rgb, kernel, kernel_size); // warm-up

        const auto gpu_start = std::chrono::high_resolution_clock::now();
        gcv::Image gpu_result = gcv::convolution_cuda_naive(rgb, kernel, kernel_size);
        const auto gpu_end = std::chrono::high_resolution_clock::now();
        const double gpu_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

        std::cout << "CPU:  " << cpu_ms << " ms\n";
        std::cout << "GPU:  " << gpu_ms << " ms (end-to-end incl. H2D/D2H transfer; warmed up)\n";
        std::cout << "Speedup: " << (cpu_ms / gpu_ms) << "x\n\n";

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
        std::cout << "  Differing bytes:     " << num_differing << " / " << n << "\n";

        // Convolution involves float accumulation across 9 (3x3) terms, so
        // allow a slightly wider tolerance than grayscale's single-sum
        // rounding, but this is still expected to be tiny (0-2), not
        // systematic drift.
        if (max_abs_error > 2) {
            std::cerr << "\nFAIL: max error exceeds the expected float-rounding "
                         "tolerance (2). Check the kernel's boundary handling "
                         "and accumulation order.\n";
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
