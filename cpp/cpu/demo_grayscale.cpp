// cpp/cpu/demo_grayscale.cpp
//
// Standalone runner for the grayscale feature only. Not wired into
// CMake/benchmark.cpp yet — that comes once we've got a few features done
// and are ready to build the real benchmark harness (FR-12). For now this
// is just: does grayscale.cpp actually work on a real image?
//
// Build (from project root):
//   g++ -std=c++17 -O2 cpp/cpu/demo_grayscale.cpp cpp/cpu/grayscale.cpp \
//       cpp/common/image_io.cpp -I. `pkg-config --cflags --libs opencv4` \
//       -o build_grayscale_demo
//
// Run:
//   ./build_grayscale_demo data/input/<your_image>.jpg data/output/grayscale.png

#include "grayscale.hpp"
#include <chrono>
#include <iostream>

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <input_image> <output_image>\n";
        return 1;
    }

    const std::string input_path = argv[1];
    const std::string output_path = argv[2];

    try
    {
        gcv::Image rgb = gcv::load_image_rgb(input_path);
        std::cout << "Loaded " << input_path << " (" << rgb.width << "x"
                  << rgb.height << ", " << rgb.channels << " channels)\n";

        const auto start = std::chrono::high_resolution_clock::now();
        gcv::Image gray = gcv::grayscale_cpu(rgb);
        const auto end = std::chrono::high_resolution_clock::now();

        const double ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "CPU grayscale: " << ms << " ms for "
                  << (rgb.width * rgb.height) << " pixels\n";

        gcv::save_image(output_path, gray);
        std::cout << "Saved " << output_path << "\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}