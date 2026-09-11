// cpp/cpu/grayscale.cpp

#include "grayscale.hpp"
#include <stdexcept>

namespace gcv
{

    Image grayscale_cpu(const Image &rgb)
    {
        if (rgb.channels != 3)
        {
            throw std::runtime_error(
                "grayscale_cpu: expected a 3-channel RGB image, got " +
                std::to_string(rgb.channels) + " channels");
        }

        Image out = make_image(rgb.width, rgb.height, 1);

        const uint8_t *src = rgb.data.get();
        uint8_t *dst = out.data.get();
        const int num_pixels = rgb.width * rgb.height;

        for (int i = 0; i < num_pixels; ++i)
        {
            const uint8_t r = src[i * 3 + 0];
            const uint8_t g = src[i * 3 + 1];
            const uint8_t b = src[i * 3 + 2];

            // Same weights the CUDA kernel will use in Phase 3, so a
            // pixel-by-pixel diff between CPU and GPU output is meaningful.
            const float gray = 0.299f * r + 0.587f * g + 0.114f * b;

            dst[i] = static_cast<uint8_t>(gray + 0.5f); // round to nearest
        }

        return out;
    }

} // namespace gcv