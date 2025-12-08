#include "block_gemm_omp.h"

#include <algorithm>
#include <cstddef>
#include <omp.h>
#include <vector>

std::vector<float> BlockGemmOMP(const std::vector<float> &a,
                                const std::vector<float> &b,
                                int n)
{
    std::size_t nn = static_cast<std::size_t>(n);
    std::vector<float> c(nn * nn, 0.0f);
    if (n <= 0)
    {
        return c;
    }

    constexpr int block_size = 64;

#pragma omp parallel for collapse(2)
    for (int bi = 0; bi < n; bi += block_size)
    {
        for (int bj = 0; bj < n; bj += block_size)
        {
            for (int bk = 0; bk < n; bk += block_size)
            {
                int i_max = std::min(bi + block_size, n);
                int j_max = std::min(bj + block_size, n);
                int k_max = std::min(bk + block_size, n);

                for (int i = bi; i < i_max; ++i)
                {
                    std::size_t row = static_cast<std::size_t>(i) * nn;
                    for (int k = bk; k < k_max; ++k)
                    {
                        float aik = a[row + static_cast<std::size_t>(k)];
                        std::size_t krow = static_cast<std::size_t>(k) * nn;
                        int j = bj;
                        for (; j + 3 < j_max; j += 4)
                        {
                            std::size_t j0 = static_cast<std::size_t>(j);
                            std::size_t j1 = j0 + 1;
                            std::size_t j2 = j0 + 2;
                            std::size_t j3 = j0 + 3;
                            c[row + j0] += aik * b[krow + j0];
                            c[row + j1] += aik * b[krow + j1];
                            c[row + j2] += aik * b[krow + j2];
                            c[row + j3] += aik * b[krow + j3];
                        }
                        for (; j < j_max; ++j)
                        {
                            std::size_t jj = static_cast<std::size_t>(j);
                            c[row + jj] += aik * b[krow + jj];
                        }
                    }
                }
            }
        }
    }

    return c;
}
