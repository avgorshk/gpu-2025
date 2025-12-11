#include "block_gemm_omp.h"

#include <omp.h>
#include <algorithm>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
    const std::vector<float>& b,
    int n)
{
    std::vector<float> c(n * n, 0.0f);

    constexpr int BS = 64;

#pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < n; ii += BS)
    {
        for (int jj = 0; jj < n; jj += BS)
        {
            for (int kk = 0; kk < n; kk += BS)
            {
                const int i_end = std::min(ii + BS, n);
                const int j_end = std::min(jj + BS, n);
                const int k_end = std::min(kk + BS, n);

                for (int i = ii; i < i_end; ++i)
                {
                    const int rowA = i * n;

                    for (int j = jj; j < j_end; ++j)
                    {
                        float acc = c[rowA + j];

                        for (int k = kk; k < k_end; ++k)
                        {
                            acc += a[rowA + k] * b[k * n + j];
                        }

                        c[rowA + j] = acc;
                    }
                }
            }
        }
    }

    return c;
}