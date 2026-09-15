// cpp/benchmark/benchmark.cpp
//
// FR-12/FR-13: runs every algorithm (CPU baseline vs its best GPU version)
// across the SRS's target resolutions/sizes, with warm-up + repeated-run
// statistics (5 warm-up, 20 measured by default, per section 31), and
// writes reports/benchmark_results.csv for Phase 7's visualization.
// Image-based algorithms sweep the four target resolutions; matrix
// multiplication sweeps matrix sizes instead, since it isn't image data.
//
// Build (from project root):
//   nvcc -std=c++17 -O2 -arch=sm_86 \
//       cpp/benchmark/benchmark.cpp \
//       cpp/cpu/grayscale.cpp cpp/cpu/inversion.cpp cpp/cpu/gaussian_blur.cpp \
//       cpp/cpu/sobel.cpp cpp/cpu/convolution.cpp cpp/cpu/histogram.cpp \
//       cpp/cpu/matrix_mul.cpp cpp/common/convolution_utils.cpp cpp/common/image_io.cpp \
//       cuda/kernels/grayscale.cu cuda/kernels/inversion.cu cuda/kernels/gaussian_blur.cu \
//       cuda/kernels/sobel.cu cuda/kernels/convolution.cu cuda/kernels/histogram.cu \
//       cuda/kernels/matrix_mul.cu cuda/optimized/tiled_convolution.cu \
//       cuda/optimized/tiled_matrix_mul.cu \
//       -I. $(pkg-config --cflags --libs opencv4) \
//       -o build_benchmark
//
// Run:
//   ./build_benchmark data/input/<your_image>.jpg reports/benchmark_results.csv
//
// Fair warning: between the resolution sweep (7 algorithms x 4
// resolutions) and the matrix multiply sweep (2 algorithms x 4 sizes,
// including a slow O(N^3) CPU baseline at N=1024), this takes a few
// minutes total, not instant.

#include "benchmark_utils.hpp"
#include "cpp/cpu/convolution.hpp"
#include "cpp/cpu/gaussian_blur.hpp"
#include "cpp/cpu/grayscale.hpp"
#include "cpp/cpu/histogram.hpp"
#include "cpp/cpu/inversion.hpp"
#include "cpp/cpu/matrix_mul.hpp"
#include "cpp/cpu/sobel.hpp"
#include "cuda/kernels/convolution.cuh"
#include "cuda/kernels/gaussian_blur.cuh"
#include "cuda/kernels/grayscale.cuh"
#include "cuda/kernels/histogram.cuh"
#include "cuda/kernels/inversion.cuh"
#include "cuda/kernels/matrix_mul.cuh"
#include "cuda/kernels/sobel.cuh"
#include "cuda/optimized/tiled_convolution.cuh"
#include "cuda/optimized/tiled_matrix_mul.cuh"
#include <functional>
#include <iostream>
#include <random>

namespace
{

    struct Resolution
    {
        int width;
        int height;
        std::string label;
    };

    // FR-13's four target resolutions.
    const std::vector<Resolution> kResolutions = {
        {640, 480, "640x480"},
        {1280, 720, "1280x720"},
        {1920, 1080, "1920x1080"},
        {3840, 2160, "3840x2160"},
    };

    void benchmark_and_write(gcv::CsvWriter &csv, const std::string &algorithm,
                             const std::string &size_label,
                             const std::function<void()> &cpu_fn,
                             const std::function<void()> &gpu_fn,
                             int warmup = 5, int measured = 20)
    {
        std::cout << "  " << algorithm << "... " << std::flush;

        const gcv::TimingStats cpu_stats = gcv::run_timed(cpu_fn, warmup, measured);
        const gcv::TimingStats gpu_stats = gcv::run_timed(gpu_fn, warmup, measured);
        const double speedup = cpu_stats.mean_ms / gpu_stats.mean_ms;

        csv.write_row({
            algorithm,
            size_label,
            std::to_string(cpu_stats.mean_ms),
            std::to_string(cpu_stats.min_ms),
            std::to_string(cpu_stats.max_ms),
            std::to_string(cpu_stats.stddev_ms),
            std::to_string(gpu_stats.mean_ms),
            std::to_string(gpu_stats.min_ms),
            std::to_string(gpu_stats.max_ms),
            std::to_string(gpu_stats.stddev_ms),
            std::to_string(speedup),
        });

        std::cout << "CPU " << cpu_stats.mean_ms << "ms, GPU " << gpu_stats.mean_ms
                  << "ms, " << speedup << "x\n";
    }

} // namespace

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <input_image> <output_csv>\n";
        return 1;
    }

    const std::string input_path = argv[1];
    const std::string csv_path = argv[2];

    try
    {
        gcv::Image base = gcv::load_image_rgb(input_path);
        std::cout << "Loaded " << input_path << " (" << base.width << "x"
                  << base.height << ") — will be resized to each target "
                  << "resolution below.\n\n";

        gcv::CsvWriter csv(csv_path);
        if (!csv.is_open())
        {
            std::cerr << "Failed to open " << csv_path << " for writing\n";
            return 1;
        }
        csv.write_header({
            "algorithm",
            "resolution",
            "cpu_mean_ms",
            "cpu_min_ms",
            "cpu_max_ms",
            "cpu_stddev_ms",
            "gpu_mean_ms",
            "gpu_min_ms",
            "gpu_max_ms",
            "gpu_stddev_ms",
            "speedup",
        });

        const std::vector<float> sharpen_kernel = gcv::kernel_sharpen();

        for (const Resolution &res : kResolutions)
        {
            std::cout << "=== " << res.label << " ===\n";
            gcv::Image img = gcv::resize_image(base, res.width, res.height);

            benchmark_and_write(csv, "grayscale", res.label, [&]()
                                { gcv::grayscale_cpu(img); }, [&]()
                                { gcv::grayscale_cuda(img); });

            benchmark_and_write(csv, "inversion", res.label, [&]()
                                { gcv::invert_cpu(img); }, [&]()
                                { gcv::invert_cuda(img); });

            benchmark_and_write(csv, "gaussian_blur", res.label, [&]()
                                { gcv::gaussian_blur_cpu(img, 5, 1.4f); }, [&]()
                                { gcv::gaussian_blur_cuda(img, 5, 1.4f); });

            benchmark_and_write(csv, "sobel", res.label, [&]()
                                { gcv::sobel_cpu(img); }, [&]()
                                { gcv::sobel_cuda(img); });

            benchmark_and_write(csv, "convolution_naive", res.label, [&]()
                                { gcv::convolution_cpu(img, sharpen_kernel, 3); }, [&]()
                                { gcv::convolution_cuda_naive(img, sharpen_kernel, 3); });

            benchmark_and_write(csv, "convolution_tiled", res.label, [&]()
                                { gcv::convolution_cpu(img, sharpen_kernel, 3); }, [&]()
                                { gcv::convolution_cuda_tiled(img, sharpen_kernel, 3); });

            benchmark_and_write(csv, "histogram", res.label, [&]()
                                { gcv::histogram_cpu(img); }, [&]()
                                { gcv::histogram_cuda(img); });

            std::cout << "\n";
        }

        // Matrix multiplication isn't image-based, so it gets its own
        // sweep over matrix sizes instead of resolutions. Naive CPU matmul
        // is O(N^3), so N=1024 already takes a noticeable fraction of a
        // second per call — fewer warmup/measured iterations at the larger
        // sizes keeps total runtime reasonable without losing the basic
        // mean/min/max/stddev picture.
        struct MatmulSize
        {
            int n;
            int warmup;
            int measured;
        };
        const std::vector<MatmulSize> matmul_sizes = {
            {128, 5, 20},
            {256, 5, 20},
            {512, 3, 10},
            {1024, 2, 5},
        };

        std::mt19937 rng(42); // same seed as the Phase 4/5 matmul demos
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        for (const MatmulSize &size : matmul_sizes)
        {
            const std::string label = std::to_string(size.n) + "x" + std::to_string(size.n);
            std::cout << "=== matmul " << label << " ===\n";

            gcv::Matrix A = gcv::make_matrix(size.n, size.n);
            gcv::Matrix B = gcv::make_matrix(size.n, size.n);
            for (size_t i = 0; i < A.size(); ++i)
                A.data[i] = dist(rng);
            for (size_t i = 0; i < B.size(); ++i)
                B.data[i] = dist(rng);

            benchmark_and_write(csv, "matmul_naive", label, [&]()
                                { gcv::matmul_cpu(A, B); }, [&]()
                                { gcv::matmul_cuda_naive(A, B); }, size.warmup, size.measured);

            benchmark_and_write(csv, "matmul_tiled", label, [&]()
                                { gcv::matmul_cpu(A, B); }, [&]()
                                { gcv::matmul_cuda_tiled(A, B); }, size.warmup, size.measured);

            std::cout << "\n";
        }

        std::cout << "Saved " << csv_path << "\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}