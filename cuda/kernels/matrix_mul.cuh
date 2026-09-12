// cuda/kernels/matrix_mul.cuh
//
// FR-10 CUDA version, naive global-memory implementation (SRS section 15).
// Each thread computes one element of C, reading its full row of A and
// column of B directly from global memory — no shared memory, no tiling.
// Same role as convolution.cu played for FR-07: this is the "before"
// measurement for the tiled/shared-memory version in Phase 5.

#pragma once

#include "../../cpp/common/matrix.hpp"

namespace gcv {

// Computes C = A x B on the GPU, naive version. Throws std::runtime_error
// if A.cols != B.rows.
Matrix matmul_cuda_naive(const Matrix& A, const Matrix& B);

} // namespace gcv
