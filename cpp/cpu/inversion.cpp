// cpp/cpu/inversion.cpp

#include "inversion.hpp"

namespace gcv
{

    Image invert_cpu(const Image &img)
    {
        Image out = make_image(img.width, img.height, img.channels);

        const uint8_t *src = img.data.get();
        uint8_t *dst = out.data.get();
        const size_t n = img.size_bytes();

        for (size_t i = 0; i < n; ++i)
        {
            // FR-04: P_new = 255 - P_old. Every byte is independent, so this
            // is even more trivially data-parallel than grayscale — the CUDA
            // kernel won't even need to know about channels or pixel structure,
            // just "1 thread = 1 byte".
            dst[i] = static_cast<uint8_t>(255 - src[i]);
        }

        return out;
    }

} // namespace gcv