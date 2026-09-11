// cpp/common/image_io.hpp
//
// Thin bridge between OpenCV (used only for reading/writing image files, per
// FR-02) and plain raw uint8_t host buffers, which is what the CPU and
// (later) CUDA implementations actually operate on. Keeping the algorithms
// working on raw pointers now means the CUDA kernels in Phase 3+ can copy
// the exact same buffers to device memory without any re-shaping.

#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace gcv
{

    // Simple owning image buffer: interleaved channels, row-major, 8-bit depth.
    // e.g. for an RGB image, pixel (x, y) channel c lives at:
    //   data[(y * width + x) * channels + c]
    struct Image
    {
        int width = 0;
        int height = 0;
        int channels = 0; // 3 for RGB, 1 for grayscale
        std::unique_ptr<uint8_t[]> data;

        size_t size_bytes() const
        {
            return static_cast<size_t>(width) * height * channels;
        }
    };

    // Reads an image file (jpg/jpeg/png/bmp/webp) into an RGB Image.
    // Throws std::runtime_error if the file can't be read or decoded.
    Image load_image_rgb(const std::string &path);

    // Writes an Image (RGB or single-channel grayscale) out to disk.
    // Format is inferred from the extension in `path`.
    // Throws std::runtime_error on failure.
    void save_image(const std::string &path, const Image &img);

    // Allocates an uninitialized Image with the given dimensions/channels.
    Image make_image(int width, int height, int channels);

} // namespace gcv