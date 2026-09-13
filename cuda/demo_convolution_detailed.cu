// cuda/demo_convolution_detailed.cu
//
// Investigates the naive-vs-tiled reversal found in the Phase 6 resolution
// sweep, using buffers allocated ONCE (not per-call) and real CUDA-event
// timing broken into H2D / kernel / D2H phases, instead of wall-clock
// around the full malloc-copy-launch-copy-free cycle. If tiled's kernel_ms
// is genuinely higher than naive's at larger resolutions, that confirms
// the 3-launches-per-channel overhead theory. If both come out similar
// once malloc/free noise is removed, the earlier reversal was mostly an
// artifact of timing methodology, not the kernels themselves.
//
// Build (from project root):
//   nvcc -std=c++17 -O2 -arch=sm_86 \
//       cuda/demo_convolution_detailed.cu \
//       cuda/kernels/convolution.cu cuda/optimized/tiled_convolution.cu \
//       cpp/cpu/convolution.cpp cpp/common/convolution_utils.cpp cpp/common/image_io.cpp \
//       -I. $(pkg-config --cflags --libs opencv4) \
//       -o build_convolution_detailed
//
// Run:
//   ./build_convolution_detailed data/input/<your_image>.jpg

#include "cpp/benchmark/benchmark_utils.hpp"
#include "cpp/cpu/convolution.hpp"
#include "cuda/kernels/convolution.cuh"
#include "cuda/optimized/tiled_convolution.cuh"
#include <iomanip>
#include <iostream>

namespace {

struct Resolution {
    int width;
    int height;
    std::string label;
};

const std::vector<Resolution> kResolutions = {
    {640, 480, "640x480"},
    {1280, 720, "1280x720"},
    {1920, 1080, "1920x1080"},
    {3840, 2160, "3840x2160"},
};

void print_breakdown(const std::string& label, const gcv::DetailedTimingSamples& samples) {
    const gcv::TimingStats h2d = gcv::compute_stats(samples.h2d_ms);
    const gcv::TimingStats kernel = gcv::compute_stats(samples.kernel_ms);
    const gcv::TimingStats d2h = gcv::compute_stats(samples.d2h_ms);
    const double total = h2d.mean_ms + kernel.mean_ms + d2h.mean_ms;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  " << label << ":\n";
    std::cout << "    H2D:    " << h2d.mean_ms << " ms\n";
    std::cout << "    Kernel: " << kernel.mean_ms << " ms\n";
    std::cout << "    D2H:    " << d2h.mean_ms << " ms\n";
    std::cout << "    Total:  " << total << " ms\n";
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_image>\n";
        return 1;
    }

    try {
        gcv::Image base = gcv::load_image_rgb(argv[1]);
        const std::vector<float> sharpen_kernel = gcv::kernel_sharpen();

        for (const Resolution& res : kResolutions) {
            std::cout << "=== " << res.label << " ===\n";
            gcv::Image img = gcv::resize_image(base, res.width, res.height);

            const gcv::DetailedTimingSamples naive =
                gcv::convolution_cuda_naive_detailed(img, sharpen_kernel, 3);
            const gcv::DetailedTimingSamples tiled =
                gcv::convolution_cuda_tiled_detailed(img, sharpen_kernel, 3);

            print_breakdown("Naive", naive);
            print_breakdown("Tiled", tiled);

            const double naive_kernel_ms = gcv::compute_stats(naive.kernel_ms).mean_ms;
            const double tiled_kernel_ms = gcv::compute_stats(tiled.kernel_ms).mean_ms;
            std::cout << "  Kernel-only tiled/naive ratio: "
                      << (tiled_kernel_ms / naive_kernel_ms)
                      << "x (< 1.0 means tiled kernel is genuinely faster)\n\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
