// cuda/optimized/tiled_matrix_mul.cuh
//
// FR-10 optimized version, shared-memory tiling (SRS section 15's "Tiled
// CUDA" column). Classic textbook GEMM tiling: each block cooperatively
// loads a TILE_DIM x TILE_DIM tile of A and B into shared memory, every
// thread in the block reuses those tiles TILE_DIM times before the next
// pair is loaded — instead of naive's every-thread-reads-global-memory-
// every-time approach. Same "before/after" role as tiled_convolution.cu.

#pragma once

#include "../../cpp/common/matrix.hpp"

namespace gcv {

// Same signature and behavior as matmul_cuda_naive — a drop-in replacement
// for direct output/timing comparison.
Matrix matmul_cuda_tiled(const Matrix& A, const Matrix& B);

} // namespace gcv
