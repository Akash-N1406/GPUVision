// cpp/cpu/demo_sobel.cpp
//
// Build (from project root):
//   g++ -std=c++17 -O2 cpp/cpu/demo_sobel.cpp cpp/cpu/sobel.cpp cpp/cpu/grayscale.cpp \
//       cpp/common/convolution_utils.cpp cpp/common/image_io.cpp -I. \
//       `pkg-config --cflags --libs opencv4` -o build_sobel_demo
//
// Run:
//   ./build_sobel_demo data/input/<your_image>.jpg data/output/sobel.png

#include "sobel.hpp"
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
        gcv::Image edges = gcv::sobel_cpu(rgb);
        const auto end = std::chrono::high_resolution_clock::now();

        const double ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "CPU Sobel: " << ms << " ms for "
                  << (rgb.width * rgb.height) << " pixels\n";

        gcv::save_image(output_path, edges);
        std::cout << "Saved " << output_path << "\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}