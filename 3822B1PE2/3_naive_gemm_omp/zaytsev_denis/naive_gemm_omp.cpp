#include "naive_gemm_omp.h"
#include <omp.h>
#include <vector>

std::vector<float> NaiveGemmOMP(const std::vector<float> &a,
                                const std::vector<float> &b,
                                int n)
{
    std::vector<float> c(n * n, 0.0f);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i)
    {
        float *c_row = &c[i * n];
        const float *a_row = &a[i * n];

        for (int k = 0; k < n; ++k)
        {
            float a_ik = a_row[k];
            const float *b_row = &b[k * n];

            for (int j = 0; j < n; ++j)
            {
                c_row[j] += a_ik * b_row[j];
            }
        }
    }

    return c;
}