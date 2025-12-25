#include "block_gemm_omp.h"
#include <vector>
#include <omp.h>
#include <cstddef>
#include <algorithm>

std::vector<float> BlockGemmOMP(const std::vector<float> &a, const std::vector<float> &b, int n)
{
    if (a.size() != static_cast<size_t>(n * n) ||
        b.size() != static_cast<size_t>(n * n))
    {
        return std::vector<float>();
    }
    std::vector<float> c(n * n, 0.0);

    const int BLOCK_SIZE = 64;

#pragma omp parallel for collapse(2) schedule(dynamic, 1)
    for (int i_out = 0; i_out < n; i_out += BLOCK_SIZE)
    {
        for (int j_out = 0; j_out < n; j_out += BLOCK_SIZE)
        {
            int i_min = std::min(i_out + BLOCK_SIZE, n);
            int j_min = std::min(j_out + BLOCK_SIZE, n);
            for (int k_out = 0; k_out < n; k_out += BLOCK_SIZE)
            {
                int k_min = std::min(k_out + BLOCK_SIZE, n);
                for (int i = i_out; i < i_min; i++)
                {
                    const float *a_local = &a[i * n];
                    for (int k = k_out; k < k_min; k++)
                    {
                        float a_ik = a_local[k];
                        const float *b_local = &b[k * n];
                        for (int j = j_out; j < j_min; j++)
                        {
                            c[i * n + j] += a_ik * b_local[j];
                        }
                    }
                }
            }
        }
    }

    return c;
}
