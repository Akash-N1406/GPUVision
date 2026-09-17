// cpp/cli/process_image.cpp
//
// Phase 8 bridge: a single CLI entry point the Django app calls via
// subprocess, instead of trying to build Python/CUDA bindings (pybind11 +
// CUDA is a much heavier, harder-to-verify lift than shelling out to an
// already-tested binary). Every algorithm below reuses the exact same
// gcv::*_cpu / gcv::*_cuda functions already verified in Phases 2-5 —
// this file adds no new algorithm logic, just a uniform argument parser,
// dispatcher, and JSON result format.
//
// Usage:
//   process_image --input <path> --output <path> --algorithm <name>
//                  --device <cpu|cuda> [--kernel-size N] [--sigma F]
//                  [--preset name]
//
// Algorithms: grayscale, inversion, gaussian_blur, sobel,
//             convolution_naive, convolution_tiled
// (histogram is excluded — its output isn't an image, so it doesn't fit
// this CLI's "process an image, save an image" contract; it would need
// its own endpoint if added to the web UI later.)
//
// Output (stdout): a single line of JSON.
//   Success: {"success":true,"algorithm":"...","device":"...",
//             "time_ms":12.34,"width":640,"height":480}
//   Failure: {"success":false,"error":"..."}
// Exit code 0 on success, 1 on failure — Django checks both the exit code
// and the JSON, since a crash before any JSON is printed still needs a
// clear failure signal.
//
// Build (from project root):
//   nvcc -std=c++17 -O2 -arch=sm_86 \
//       cpp/cli/process_image.cpp \
//       cpp/cpu/grayscale.cpp cpp/cpu/inversion.cpp cpp/cpu/gaussian_blur.cpp \
//       cpp/cpu/sobel.cpp cpp/cpu/convolution.cpp cpp/common/convolution_utils.cpp \
//       cpp/common/image_io.cpp \
//       cuda/kernels/grayscale.cu cuda/kernels/inversion.cu cuda/kernels/gaussian_blur.cu \
//       cuda/kernels/sobel.cu cuda/kernels/convolution.cu cuda/optimized/tiled_convolution.cu \
//       -I. $(pkg-config --cflags --libs opencv4) \
//       -o process_image

#include "cpp/cpu/convolution.hpp"
#include "cpp/cpu/gaussian_blur.hpp"
#include "cpp/cpu/grayscale.hpp"
#include "cpp/cpu/inversion.hpp"
#include "cpp/cpu/sobel.hpp"
#include "cuda/kernels/convolution.cuh"
#include "cuda/kernels/gaussian_blur.cuh"
#include "cuda/kernels/grayscale.cuh"
#include "cuda/kernels/inversion.cuh"
#include "cuda/kernels/sobel.cuh"
#include "cuda/optimized/tiled_convolution.cuh"
#include <chrono>
#include <iostream>
#include <map>
#include <string>

namespace
{

    struct Args
    {
        std::string input;
        std::string output;
        std::string algorithm;
        std::string device = "cpu";
        int kernel_size = 5;
        float sigma = 1.4f;
        std::string preset = "sharpen";
    };

    // Minimal manual flag parser — no external dependency needed for a handful
    // of --flag value pairs.
    Args parse_args(int argc, char **argv)
    {
        Args args;
        std::map<std::string, std::string> flags;
        for (int i = 1; i + 1 < argc; i += 2)
        {
            std::string key = argv[i];
            if (key.rfind("--", 0) == 0)
            {
                flags[key.substr(2)] = argv[i + 1];
            }
        }

        if (flags.count("input"))
            args.input = flags["input"];
        if (flags.count("output"))
            args.output = flags["output"];
        if (flags.count("algorithm"))
            args.algorithm = flags["algorithm"];
        if (flags.count("device"))
            args.device = flags["device"];
        if (flags.count("kernel-size"))
            args.kernel_size = std::stoi(flags["kernel-size"]);
        if (flags.count("sigma"))
            args.sigma = std::stof(flags["sigma"]);
        if (flags.count("preset"))
            args.preset = flags["preset"];

        return args;
    }

    // Escapes the handful of characters that would break a hand-rolled JSON
    // string literal (a full JSON library would be overkill for one error
    // message field).
    std::string json_escape(const std::string &s)
    {
        std::string out;
        out.reserve(s.size());
        for (char c : s)
        {
            if (c == '"' || c == '\\')
                out += '\\';
            out += c;
        }
        return out;
    }

    void print_success(const std::string &algorithm, const std::string &device,
                       double time_ms, int width, int height)
    {
        std::cout << "{\"success\":true,\"algorithm\":\"" << algorithm
                  << "\",\"device\":\"" << device << "\",\"time_ms\":" << time_ms
                  << ",\"width\":" << width << ",\"height\":" << height << "}\n";
    }

    void print_failure(const std::string &error)
    {
        std::cout << "{\"success\":false,\"error\":\"" << json_escape(error) << "\"}\n";
    }

} // namespace

int main(int argc, char **argv)
{
    Args args = parse_args(argc, argv);

    if (args.input.empty() || args.output.empty() || args.algorithm.empty())
    {
        print_failure("missing required argument: --input, --output, and --algorithm are all required");
        return 1;
    }
    if (args.device != "cpu" && args.device != "cuda")
    {
        print_failure("--device must be 'cpu' or 'cuda', got '" + args.device + "'");
        return 1;
    }

    try
    {
        gcv::Image input = gcv::load_image_rgb(args.input);
        gcv::Image result;

        // A single CLI invocation is a fresh process, so the very first
        // CUDA API call always pays CUDA's one-time context-initialization
        // cost (can be several hundred ms) — every benchmark demo in this
        // project absorbs that with a warm-up call before timing, and this
        // CLI needs the same treatment. Without it, a web-UI user clicking
        // "process with GPU" would see a wildly inflated number that
        // contradicts every other speedup figure in the project. CPU has
        // no equivalent cold-start cost, so it skips this.
        auto run_once = [&]() -> gcv::Image
        {
            if (args.algorithm == "grayscale")
            {
                return (args.device == "cpu") ? gcv::grayscale_cpu(input)
                                              : gcv::grayscale_cuda(input);
            }
            else if (args.algorithm == "inversion")
            {
                return (args.device == "cpu") ? gcv::invert_cpu(input)
                                              : gcv::invert_cuda(input);
            }
            else if (args.algorithm == "gaussian_blur")
            {
                return (args.device == "cpu")
                           ? gcv::gaussian_blur_cpu(input, args.kernel_size, args.sigma)
                           : gcv::gaussian_blur_cuda(input, args.kernel_size, args.sigma);
            }
            else if (args.algorithm == "sobel")
            {
                return (args.device == "cpu") ? gcv::sobel_cpu(input)
                                              : gcv::sobel_cuda(input);
            }
            else if (args.algorithm == "convolution_naive")
            {
                const std::vector<float> kernel = gcv::get_preset_kernel(args.preset);
                return (args.device == "cpu")
                           ? gcv::convolution_cpu(input, kernel, 3)
                           : gcv::convolution_cuda_naive(input, kernel, 3);
            }
            else if (args.algorithm == "convolution_tiled")
            {
                const std::vector<float> kernel = gcv::get_preset_kernel(args.preset);
                return (args.device == "cpu")
                           ? gcv::convolution_cpu(input, kernel, 3)
                           : gcv::convolution_cuda_tiled(input, kernel, 3);
            }
            throw std::runtime_error(
                "unknown algorithm: '" + args.algorithm +
                "' (expected one of: grayscale, inversion, gaussian_blur, "
                "sobel, convolution_naive, convolution_tiled)");
        };

        if (args.device == "cuda")
        {
            run_once(); // warm-up, result discarded — absorbs context init
        }

        const auto start = std::chrono::high_resolution_clock::now();
        result = run_once();
        const auto end = std::chrono::high_resolution_clock::now();
        const double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        gcv::save_image(args.output, result);
        print_success(args.algorithm, args.device, time_ms, result.width, result.height);
    }
    catch (const std::exception &e)
    {
        print_failure(e.what());
        return 1;
    }

    return 0;
}