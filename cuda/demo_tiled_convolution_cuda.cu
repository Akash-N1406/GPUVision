// cuda/demo_tiled_convolution_cuda.cu
//
// The key comparison for this whole project: CPU baseline vs naive CUDA
// (global memory only) vs tiled CUDA (shared memory). All three should
// produce matching output; the interesting number is the naive-vs-tiled
// speedup, not just CPU-vs-GPU.
//
// Build (from project root):
//   nvcc -std=c++17 -O2 -arch=sm_86 \
//       cuda/demo_tiled_convolution_cuda.cu \
//       cuda/kernels/convolution.cu cuda/optimized/tiled_convolution.cu \
//       cpp/cpu/convolution.cpp cpp/common/convolution_utils.cpp cpp/common/image_io.cpp \
//       -I. $(pkg-config --cflags --libs opencv4) \
//       -o build_tiled_convolution_demo
//
// Run:
//   ./build_tiled_convolution_demo data/input/<your_image>.jpg data/output/tiled_gpu.png sharpen

#include "cpp/cpu/convolution.hpp"
#include "cuda/kernels/convolution.cuh"
#include "cuda/optimized/tiled_convolution.cuh"
#include <chrono>
#include <cmath>
#include <iostream>

namespace {

// Returns {max_abs_error, num_differing}.
std::pair<int, int> compare_images(const gcv::Image& a, const gcv::Image& b) {
    const size_t n = a.size_bytes();
    int max_abs_error = 0;
    int num_differing = 0;
    for (size_t i = 0; i < n; ++i) {
        const int diff = std::abs(static_cast<int>(a.data[i]) - static_cast<int>(b.data[i]));
        if (diff > 0) num_differing++;
        if (diff > max_abs_error) max_abs_error = diff;
    }
    return {max_abs_error, num_differing};
}

} // namespace

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

        // CPU baseline
        const auto cpu_start = std::chrono::high_resolution_clock::now();
        gcv::Image cpu_result = gcv::convolution_cpu(rgb, kernel, kernel_size);
        const auto cpu_end = std::chrono::high_resolution_clock::now();
        const double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

        // Naive CUDA (global memory only)
        gcv::convolution_cuda_naive(rgb, kernel, kernel_size); // warm-up
        const auto naive_start = std::chrono::high_resolution_clock::now();
        gcv::Image naive_result = gcv::convolution_cuda_naive(rgb, kernel, kernel_size);
        const auto naive_end = std::chrono::high_resolution_clock::now();
        const double naive_ms = std::chrono::duration<double, std::milli>(naive_end - naive_start).count();

        // Tiled CUDA (shared memory)
        gcv::convolution_cuda_tiled(rgb, kernel, kernel_size); // warm-up
        const auto tiled_start = std::chrono::high_resolution_clock::now();
        gcv::Image tiled_result = gcv::convolution_cuda_tiled(rgb, kernel, kernel_size);
        const auto tiled_end = std::chrono::high_resolution_clock::now();
        const double tiled_ms = std::chrono::duration<double, std::milli>(tiled_end - tiled_start).count();

        std::cout << "CPU:         " << cpu_ms << " ms\n";
        std::cout << "Naive CUDA:  " << naive_ms << " ms  (speedup vs CPU: "
                  << (cpu_ms / naive_ms) << "x)\n";
        std::cout << "Tiled CUDA:  " << tiled_ms << " ms  (speedup vs CPU: "
                  << (cpu_ms / tiled_ms) << "x)\n";
        std::cout << "Tiled vs Naive speedup: " << (naive_ms / tiled_ms) << "x\n\n";

        // Correctness: all three must agree.
        const auto [naive_max_err, naive_diff] = compare_images(cpu_result, naive_result);
        const auto [tiled_max_err, tiled_diff] = compare_images(cpu_result, tiled_result);
        const auto [cross_max_err, cross_diff] = compare_images(naive_result, tiled_result);

        const size_t n = cpu_result.size_bytes();
        std::cout << "Correctness (vs CPU baseline):\n";
        std::cout << "  Naive CUDA: max error " << naive_max_err << ", "
                  << naive_diff << "/" << n << " bytes differ\n";
        std::cout << "  Tiled CUDA: max error " << tiled_max_err << ", "
                  << tiled_diff << "/" << n << " bytes differ\n";
        std::cout << "  Naive vs Tiled (should match each other too): max error "
                  << cross_max_err << ", " << cross_diff << "/" << n << " bytes differ\n";

        constexpr int TOLERANCE = 2;
        if (naive_max_err > TOLERANCE || tiled_max_err > TOLERANCE || cross_max_err > TOLERANCE) {
            std::cerr << "\nFAIL: at least one comparison exceeds tolerance ("
                      << TOLERANCE << "). If naive passes but tiled fails, the bug "
                      << "is almost certainly in the shared-memory tile/halo loading "
                      << "logic, not the math.\n";
            return 1;
        }
        std::cout << "\nPASS: CPU, naive CUDA, and tiled CUDA all agree within tolerance.\n";

        gcv::save_image(output_path, tiled_result);
        std::cout << "Saved " << output_path << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
