#include "block_gemm_omp.h"
#include <omp.h>
#include <algorithm>

std::vector<float> BlockGemmOMP(const std::vector<float> &a,
                                const std::vector<float> &b,
                                int n)
{
    std::vector<float> c(n * n, 0.0f);

    const int BLOCK_SIZE = 32;

#pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < n; ii += BLOCK_SIZE)
    {
        for (int jj = 0; jj < n; jj += BLOCK_SIZE)
        {
            for (int kk = 0; kk < n; kk += BLOCK_SIZE)
            {
                int i_max = std::min(ii + BLOCK_SIZE, n);
                int j_max = std::min(jj + BLOCK_SIZE, n);
                int k_max = std::min(kk + BLOCK_SIZE, n);

                for (int i = ii; i < i_max; ++i)
                {
                    for (int k = kk; k < k_max; ++k)
                    {
                        float a_ik = a[i * n + k];
#pragma omp simd
                        for (int j = jj; j < j_max; ++j)
                        {
                            c[i * n + j] += a_ik * b[k * n + j];
                        }
                    }
                }
            }
        }
    }

    return c;
}