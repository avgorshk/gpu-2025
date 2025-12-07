#include "naive_gemm_omp.h"

#include <cstddef>
#include <omp.h>
#include <vector>

std::vector<float> NaiveGemmOMP(const std::vector<float> &a,
                                const std::vector<float> &b,
                                int n)
{
    const std::size_t nn = static_cast<std::size_t>(n);
    std::vector<float> c(nn * nn);
    if (nn == 0)
    {
        return c;
    }

    std::vector<float> bt(nn * nn);
    for (std::size_t i = 0; i < nn; ++i)
    {
        const std::size_t row = i * nn;
        for (std::size_t j = 0; j < nn; ++j)
        {
            bt[j * nn + i] = b[row + j];
        }
    }

#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            const std::size_t row = static_cast<std::size_t>(i) * nn;
            const std::size_t col = static_cast<std::size_t>(j) * nn;
            float sum = 0.0f;
            std::size_t k = 0;
            for (; k + 3 < nn; k += 4)
            {
                sum += a[row + k] * bt[col + k];
                sum += a[row + k + 1] * bt[col + k + 1];
                sum += a[row + k + 2] * bt[col + k + 2];
                sum += a[row + k + 3] * bt[col + k + 3];
            }
            for (; k < nn; ++k)
            {
                sum += a[row + k] * bt[col + k];
            }
            c[row + static_cast<std::size_t>(j)] = sum;
        }
    }

    return c;
}
