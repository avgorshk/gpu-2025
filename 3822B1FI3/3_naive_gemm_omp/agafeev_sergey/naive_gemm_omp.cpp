#include "naive_gemm_omp.h"
#include <omp.h>
#include <chrono>

std::vector<float> NaiveGemmOMP(const std::vector<float> &a,
                                const std::vector<float> &b,
                                int sz)
{
    std::vector<float> res(sz * sz, 0.0f);

#pragma omp parallel for
    for (int i = 0; i < sz; ++i)
    {
        int precalc = i * sz;
        for (int j = 0; j < sz; ++j)
        {
            for (int k = 0; k < sz; ++k)
            {
                res[precalc + j] += a[precalc + k] * b[k * sz + j];
            }
        }
    }
    return res;
}