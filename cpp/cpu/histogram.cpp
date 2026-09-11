// cpp/cpu/histogram.cpp

#include "histogram.hpp"
#include "grayscale.hpp"
#include <stdexcept>

namespace gcv
{

    Histogram histogram_cpu(const Image &img)
    {
        // Same move-only Image gotcha as sobel.cpp: can't use a ternary to pick
        // between `img` and grayscale_cpu(img), so hold an optional owned copy.
        Image gray_owned;
        const Image *gray_ptr = &img;
        if (img.channels != 1)
        {
            gray_owned = grayscale_cpu(img);
            gray_ptr = &gray_owned;
        }
        const Image &gray = *gray_ptr;
        if (gray.channels != 1)
        {
            throw std::runtime_error("histogram_cpu: expected a 1 or 3 channel image");
        }

        Histogram hist{}; // zero-initialized
        const uint8_t *data = gray.data.get();
        const size_t n = gray.size_bytes();

        for (size_t i = 0; i < n; ++i)
        {
            hist[data[i]]++;
        }

        return hist;
    }

} // namespace gcv