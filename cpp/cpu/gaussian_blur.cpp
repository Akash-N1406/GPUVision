// cpp/cpu/gaussian_blur.cpp

#include "gaussian_blur.hpp"
#include "../common/convolution_utils.hpp"
#include <cmath>
#include <stdexcept>

namespace gcv
{

    namespace
    {

        std::vector<float> make_gaussian_kernel(int kernel_size, float sigma)
        {
            if (kernel_size % 2 == 0)
            {
                throw std::runtime_error("make_gaussian_kernel: kernel_size must be odd");
            }

            std::vector<float> kernel(kernel_size * kernel_size);
            const int radius = kernel_size / 2;
            float sum = 0.0f;

            for (int y = -radius; y <= radius; ++y)
            {
                for (int x = -radius; x <= radius; ++x)
                {
                    const float value =
                        std::exp(-(x * x + y * y) / (2.0f * sigma * sigma));
                    kernel[(y + radius) * kernel_size + (x + radius)] = value;
                    sum += value;
                }
            }

            // Normalize so the kernel sums to 1 — otherwise blurred images would
            // get systematically brighter or darker.
            for (float &v : kernel)
            {
                v /= sum;
            }

            return kernel;
        }

    } // namespace

    Image gaussian_blur_cpu(const Image &img, int kernel_size, float sigma)
    {
        const std::vector<float> kernel = make_gaussian_kernel(kernel_size, sigma);
        return apply_kernel_cpu(img, kernel, kernel_size);
    }

} // namespace gcv