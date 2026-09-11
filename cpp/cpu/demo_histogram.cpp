// cpp/cpu/demo_histogram.cpp
//
// Build (from project root):
//   g++ -std=c++17 -O2 cpp/cpu/demo_histogram.cpp cpp/cpu/histogram.cpp cpp/cpu/grayscale.cpp \
//       cpp/common/image_io.cpp -I. `pkg-config --cflags --libs opencv4` \
//       -o build_histogram_demo
//
// Run:
//   ./build_histogram_demo data/input/<your_image>.jpg reports/histogram.csv

#include "histogram.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

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
        gcv::Image rgb = gcv::load_image_rgb(input_path);
        std::cout << "Loaded " << input_path << " (" << rgb.width << "x"
                  << rgb.height << ", " << rgb.channels << " channels)\n";

        const auto start = std::chrono::high_resolution_clock::now();
        gcv::Histogram hist = gcv::histogram_cpu(rgb);
        const auto end = std::chrono::high_resolution_clock::now();

        const double ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "CPU histogram: " << ms << " ms\n";

        // Sanity check: bin counts must sum to total pixel count. This is
        // exactly the kind of check that matters once the CUDA version uses
        // atomics — a race condition would silently under-count here.
        long long total = 0;
        for (int count : hist)
            total += count;
        const long long expected = static_cast<long long>(rgb.width) * rgb.height;
        std::cout << "Sanity check: sum(bins)=" << total << " vs pixels="
                  << expected << " -> " << (total == expected ? "PASS" : "FAIL") << "\n";

        // Small ASCII visualization (20 buckets of 12-13 intensity values each).
        const int num_buckets = 20;
        const int bucket_width = 256 / num_buckets;
        int max_bucket_count = 0;
        std::vector<int> buckets(num_buckets, 0);
        for (int i = 0; i < 256; ++i)
        {
            buckets[std::min(i / bucket_width, num_buckets - 1)] += hist[i];
        }
        max_bucket_count = *std::max_element(buckets.begin(), buckets.end());

        std::cout << "\nIntensity distribution:\n";
        for (int b = 0; b < num_buckets; ++b)
        {
            const int bar_len = max_bucket_count > 0
                                    ? (buckets[b] * 40) / max_bucket_count
                                    : 0;
            std::cout << (b * bucket_width) << "-" << (b * bucket_width + bucket_width - 1)
                      << "\t" << std::string(bar_len, '#') << "\n";
        }

        // CSV export for Phase 7 visualization.
        std::ofstream csv(csv_path);
        csv << "intensity,count\n";
        for (int i = 0; i < 256; ++i)
        {
            csv << i << "," << hist[i] << "\n";
        }
        std::cout << "\nSaved " << csv_path << "\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}