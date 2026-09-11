// cpp/common/convolution_utils.cpp

#include "convolution_utils.hpp"
#include <algorithm>
#include <stdexcept>

namespace gcv
{

    namespace
    {

        inline int clamp_coord(int v, int lo, int hi)
        {
            return std::max(lo, std::min(v, hi));
        }

        inline uint8_t clamp_to_byte(float v)
        {
            if (v < 0.0f)
                return 0;
            if (v > 255.0f)
                return 255;
            return static_cast<uint8_t>(v + 0.5f);
        }

    } // namespace

    namespace
    {

        void validate_kernel(const std::vector<float> &kernel, int kernel_size,
                             const char *caller)
        {
            if (kernel_size % 2 == 0)
            {
                throw std::runtime_error(std::string(caller) + ": kernel_size must be odd");
            }
            if (static_cast<int>(kernel.size()) != kernel_size * kernel_size)
            {
                throw std::runtime_error(
                    std::string(caller) + ": kernel size mismatch (expected " +
                    std::to_string(kernel_size * kernel_size) + " elements, got " +
                    std::to_string(kernel.size()) + ")");
            }
        }

    } // anonymous namespace

    std::vector<float> apply_kernel_raw_cpu(const Image &img,
                                            const std::vector<float> &kernel,
                                            int kernel_size)
    {
        validate_kernel(kernel, kernel_size, "apply_kernel_raw_cpu");

        const uint8_t *src = img.data.get();
        const int radius = kernel_size / 2;
        const int W = img.width;
        const int H = img.height;
        const int C = img.channels;

        std::vector<float> out(static_cast<size_t>(W) * H * C, 0.0f);

        for (int y = 0; y < H; ++y)
        {
            for (int x = 0; x < W; ++x)
            {
                for (int c = 0; c < C; ++c)
                {
                    float sum = 0.0f;

                    for (int ky = -radius; ky <= radius; ++ky)
                    {
                        const int sy = clamp_coord(y + ky, 0, H - 1);
                        for (int kx = -radius; kx <= radius; ++kx)
                        {
                            const int sx = clamp_coord(x + kx, 0, W - 1);
                            const float weight =
                                kernel[(ky + radius) * kernel_size + (kx + radius)];
                            sum += weight * src[(sy * W + sx) * C + c];
                        }
                    }

                    out[(y * W + x) * C + c] = sum;
                }
            }
        }

        return out;
    }

    Image apply_kernel_cpu(const Image &img, const std::vector<float> &kernel,
                           int kernel_size)
    {
        validate_kernel(kernel, kernel_size, "apply_kernel_cpu");

        const std::vector<float> raw = apply_kernel_raw_cpu(img, kernel, kernel_size);

        Image out = make_image(img.width, img.height, img.channels);
        uint8_t *dst = out.data.get();
        for (size_t i = 0; i < raw.size(); ++i)
        {
            dst[i] = clamp_to_byte(raw[i]);
        }

        return out;
    }

} // namespace gcv