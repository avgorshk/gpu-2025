#include "block_gemm_omp.h"
#include <vector>
#include <omp.h>

std::vector<float> BlockGemmOMP(const std::vector<float> &a, const std::vector<float> &b, int n)
{

    std::vector<float> res(n * n, 0.0);
    int block = 2;
    for (int i = 1; i < n; i = i * 2)
    {
        if (i >= n / 8)
        {
            block = i;
            break;
        }
    }

    omp_set_num_threads(4);
#pragma omp parallel for collapse(2) schedule(dynamic, 1)
    for (int i = 0; i < n; i += block)
    {
        for (int j = 0; j < n; j += block)
        {
            int x_cursor = std::min(i + block, n);
            int y_cursor = std::min(j + block, n);
            for (int k = 0; k < n; k += block)
            {
                int z_cursor = std::min(k + block, n);
                for (int L = i; L < x_cursor; L++)
                {
                    const float *a_local = &a[L * n];
                    for (int m = k; m < z_cursor; m++)
                    {
                        const float *b_local = &b[m * n];
                        for (int x = j; x < y_cursor; x++)
                        {
                            res[L * n + x] += a_local[m] * b_local[x];
                        }
                    }
                }
            }
        }
    }

    return res;
}
