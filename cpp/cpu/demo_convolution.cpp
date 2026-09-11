// cpp/cpu/demo_convolution.cpp
//
// Build (from project root):
//   g++ -std=c++17 -O2 cpp/cpu/demo_convolution.cpp cpp/cpu/convolution.cpp \
//       cpp/common/convolution_utils.cpp cpp/common/image_io.cpp -I. \
//       `pkg-config --cflags --libs opencv4` -o build_convolution_demo
//
// Run:
//   ./build_convolution_demo data/input/<your_image>.jpg data/output/conv.png sharpen
//   (preset: sharpen | edge | emboss | box_blur)

#include "convolution.hpp"
#include <chrono>
#include <iostream>

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <input_image> <output_image> <preset: sharpen|edge|emboss|box_blur>\n";
        return 1;
    }

    const std::string input_path = argv[1];
    const std::string output_path = argv[2];
    const std::string preset = argv[3];

    try
    {
        const std::vector<float> kernel = gcv::get_preset_kernel(preset);

        gcv::Image rgb = gcv::load_image_rgb(input_path);
        std::cout << "Loaded " << input_path << " (" << rgb.width << "x"
                  << rgb.height << ", " << rgb.channels << " channels)\n";
        std::cout << "Preset: " << preset << "\n";

        const auto start = std::chrono::high_resolution_clock::now();
        gcv::Image result = gcv::convolution_cpu(rgb, kernel, 3);
        const auto end = std::chrono::high_resolution_clock::now();

        const double ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "CPU convolution (" << preset << "): " << ms << " ms for "
                  << (rgb.width * rgb.height) << " pixels\n";

        gcv::save_image(output_path, result);
        std::cout << "Saved " << output_path << "\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}