// cpp/cpu/demo_gaussian_blur.cpp
//
// Build (from project root):
//   g++ -std=c++17 -O2 cpp/cpu/demo_gaussian_blur.cpp cpp/cpu/gaussian_blur.cpp \
//       cpp/common/convolution_utils.cpp cpp/common/image_io.cpp -I. \
//       `pkg-config --cflags --libs opencv4` -o build_blur_demo
//
// Run:
//   ./build_blur_demo data/input/<your_image>.jpg data/output/blurred.png [kernel_size] [sigma]

#include "gaussian_blur.hpp"
#include <chrono>
#include <iostream>

int main(int argc, char **argv)
{
    if (argc < 3 || argc > 5)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <input_image> <output_image> [kernel_size=5] [sigma=1.4]\n";
        return 1;
    }

    const std::string input_path = argv[1];
    const std::string output_path = argv[2];
    const int kernel_size = argc > 3 ? std::stoi(argv[3]) : 5;
    const float sigma = argc > 4 ? std::stof(argv[4]) : 1.4f;

    try
    {
        gcv::Image rgb = gcv::load_image_rgb(input_path);
        std::cout << "Loaded " << input_path << " (" << rgb.width << "x"
                  << rgb.height << ", " << rgb.channels << " channels)\n";
        std::cout << "Kernel: " << kernel_size << "x" << kernel_size
                  << ", sigma=" << sigma << "\n";

        const auto start = std::chrono::high_resolution_clock::now();
        gcv::Image blurred = gcv::gaussian_blur_cpu(rgb, kernel_size, sigma);
        const auto end = std::chrono::high_resolution_clock::now();

        const double ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "CPU Gaussian blur: " << ms << " ms for "
                  << (rgb.width * rgb.height) << " pixels\n";

        gcv::save_image(output_path, blurred);
        std::cout << "Saved " << output_path << "\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}