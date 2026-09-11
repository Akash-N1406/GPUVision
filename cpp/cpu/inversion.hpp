// cpp/cpu/inversion.hpp
//
// FR-04: Image inversion, CPU baseline.
// P_new = 255 - P_old, per SRS section 9.

#pragma once

#include "../common/image_io.hpp"

namespace gcv
{

    // Inverts every channel of every pixel. Works on RGB or single-channel
    // images alike since the transform is per-byte, not per-pixel-semantics —
    // unlike grayscale, there's no channel mixing here.
    Image invert_cpu(const Image &img);

} // namespace gcv