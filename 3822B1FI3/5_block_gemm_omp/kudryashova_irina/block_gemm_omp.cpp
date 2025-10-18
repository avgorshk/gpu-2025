#include "block_gemm_omp.h"
#include <algorithm>
#include <omp.h>

std::vector<float> BlockGemmOMP(const std::vector<float> &a, const std::vector<float> &b, int n)
{
    const int block_size = 64;
    std::vector<float> result(n * n, 0.0f);

#pragma omp parallel for collapse(2) schedule(static)
    for (int block_row = 0; block_row < n; block_row += block_size)
    {
        for (int block_col = 0; block_col < n; block_col += block_size)
        {

            const int i_end = std::min(block_row + block_size, n);
            const int j_end = std::min(block_col + block_size, n);

            for (int block_inner = 0; block_inner < n; block_inner += block_size)
            {
                const int k_end = std::min(block_inner + block_size, n);

                for (int i = block_row; i < i_end; ++i)
                {
                    const float *a_row_base = &a[i * n];
                    float *c_row = &result[i * n];

                    for (int j = block_col; j < j_end; ++j)
                    {
                        float sum = c_row[j];
#pragma omp simd reduction(+ : sum)
                        for (int k = block_inner; k < k_end; ++k)
                        {
                            sum += a_row_base[k] * b[k * n + j];
                        }

                        c_row[j] = sum;
                    }
                }
            }
        }
    }
    return result;
}
