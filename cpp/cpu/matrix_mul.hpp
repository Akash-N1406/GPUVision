// cpp/cpu/matrix_mul.hpp
//
// FR-10: matrix multiplication, CPU baseline.
// C = A x B via the straightforward triple loop from SRS section 15 —
// deliberately naive (no blocking/cache tiling), since the point of this
// feature is to be the fair, unoptimized baseline that naive CUDA and then
// tiled/shared-memory CUDA get benchmarked against in Phase 5/6.

#pragma once

#include "../common/matrix.hpp"

namespace gcv
{

    // Computes C = A x B. Throws std::runtime_error if A.cols != B.rows.
    Matrix matmul_cpu(const Matrix &A, const Matrix &B);

} // namespace gcv