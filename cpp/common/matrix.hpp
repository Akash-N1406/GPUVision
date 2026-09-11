// cpp/common/matrix.hpp
//
// Simple owning row-major matrix of floats, used by FR-10 (matrix
// multiplication). Kept separate from image_io.hpp's Image type since
// matmul isn't image data — but it follows the same "raw host buffer,
// ready to copy to device memory" shape so the CUDA kernels in Phase 4
// (naive) and Phase 5 (tiled) can reuse it directly.

#pragma once

#include <memory>
#include <stdexcept>

namespace gcv
{

    struct Matrix
    {
        int rows = 0;
        int cols = 0;
        std::unique_ptr<float[]> data;

        size_t size() const { return static_cast<size_t>(rows) * cols; }

        float &at(int r, int c) { return data[static_cast<size_t>(r) * cols + c]; }
        float at(int r, int c) const { return data[static_cast<size_t>(r) * cols + c]; }
    };

    inline Matrix make_matrix(int rows, int cols)
    {
        Matrix m;
        m.rows = rows;
        m.cols = cols;
        m.data = std::make_unique<float[]>(m.size());
        return m;
    }

} // namespace gcv