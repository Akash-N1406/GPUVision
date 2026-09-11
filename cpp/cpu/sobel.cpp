// cpp/cpu/sobel.cpp

#include "sobel.hpp"
#include "grayscale.hpp"
#include "../common/convolution_utils.hpp"
#include <cmath>
#include <stdexcept>

namespace gcv
{

    Image sobel_cpu(const Image &img)
    {
        // Sobel is conventionally computed on a single intensity channel —
        // running it independently per RGB channel and recombining produces
        // noisier, harder-to-interpret edges, so we normalize to grayscale
        // first, same as the CUDA kernel will do in Phase 3.
        // Image is move-only (owns a unique_ptr buffer), so we can't use a
        // ternary here — hold an optional owned copy and point at whichever
        // one is live.
        Image gray_owned;
        const Image *gray_ptr = &img;
        if (img.channels != 1)
        {
            gray_owned = grayscale_cpu(img);
            gray_ptr = &gray_owned;
        }
        const Image &gray_ref = *gray_ptr;
        if (gray_ref.channels != 1)
        {
            throw std::runtime_error("sobel_cpu: expected a 1 or 3 channel image");
        }

        // Gx and Gy kernels, per SRS section 11.
        static const std::vector<float> kx = {
            -1,
            0,
            1,
            -2,
            0,
            2,
            -1,
            0,
            1,
        };
        static const std::vector<float> ky = {
            -1,
            -2,
            -1,
            0,
            0,
            0,
            1,
            2,
            1,
        };

        const std::vector<float> gx = apply_kernel_raw_cpu(gray_ref, kx, 3);
        const std::vector<float> gy = apply_kernel_raw_cpu(gray_ref, ky, 3);

        Image out = make_image(gray_ref.width, gray_ref.height, 1);
        uint8_t *dst = out.data.get();

        for (size_t i = 0; i < gx.size(); ++i)
        {
            const float magnitude = std::sqrt(gx[i] * gx[i] + gy[i] * gy[i]);
            dst[i] = static_cast<uint8_t>(
                std::min(255.0f, std::max(0.0f, magnitude + 0.5f)));
        }

        return out;
    }

} // namespace gcv