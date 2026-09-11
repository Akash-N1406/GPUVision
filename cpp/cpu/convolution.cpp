// cpp/cpu/convolution.cpp

#include "convolution.hpp"
#include "../common/convolution_utils.hpp"
#include <stdexcept>

namespace gcv
{

    Image convolution_cpu(const Image &img, const std::vector<float> &kernel,
                          int kernel_size)
    {
        return apply_kernel_cpu(img, kernel, kernel_size);
    }

    std::vector<float> kernel_sharpen()
    {
        return {
            0,
            -1,
            0,
            -1,
            5,
            -1,
            0,
            -1,
            0,
        };
    }

    std::vector<float> kernel_edge_detect()
    {
        return {
            0,
            -1,
            0,
            -1,
            4,
            -1,
            0,
            -1,
            0,
        };
    }

    std::vector<float> kernel_emboss()
    {
        return {
            -2,
            -1,
            0,
            -1,
            1,
            1,
            0,
            1,
            2,
        };
    }

    std::vector<float> kernel_box_blur_3x3()
    {
        const float w = 1.0f / 9.0f;
        return {
            w,
            w,
            w,
            w,
            w,
            w,
            w,
            w,
            w,
        };
    }

    std::vector<float> get_preset_kernel(const std::string &name)
    {
        if (name == "sharpen")
            return kernel_sharpen();
        if (name == "edge")
            return kernel_edge_detect();
        if (name == "emboss")
            return kernel_emboss();
        if (name == "box_blur")
            return kernel_box_blur_3x3();
        throw std::runtime_error(
            "get_preset_kernel: unknown preset '" + name +
            "' (expected one of: sharpen, edge, emboss, box_blur)");
    }

} // namespace gcv