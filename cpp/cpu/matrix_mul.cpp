// cpp/cpu/matrix_mul.cpp

#include "matrix_mul.hpp"

namespace gcv
{

    Matrix matmul_cpu(const Matrix &A, const Matrix &B)
    {
        if (A.cols != B.rows)
        {
            throw std::runtime_error(
                "matmul_cpu: dimension mismatch (A is " + std::to_string(A.rows) +
                "x" + std::to_string(A.cols) + ", B is " + std::to_string(B.rows) +
                "x" + std::to_string(B.cols) + ")");
        }

        Matrix C = make_matrix(A.rows, B.cols);

        for (int i = 0; i < A.rows; ++i)
        {
            for (int j = 0; j < B.cols; ++j)
            {
                float sum = 0.0f;
                for (int k = 0; k < A.cols; ++k)
                {
                    sum += A.at(i, k) * B.at(k, j);
                }
                C.at(i, j) = sum;
            }
        }

        return C;
    }

} // namespace gcv