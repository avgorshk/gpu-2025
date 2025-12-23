#include "block_gemm_omp.h"
#include <omp.h>
#include <cstddef>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    const std::size_t nn = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    if (a.size() != nn || b.size() != nn) {
        return {};
    }

    std::vector<float> c(nn, 0.0f);

    const int BS = 64;

    #pragma omp parallel for collapse(2)
    for (int ii = 0; ii < n; ii += BS) {
        for (int jj = 0; jj < n; jj += BS) {

            for (int kk = 0; kk < n; kk += BS) {

                int i_max = ii + BS;
                int j_max = jj + BS;
                int k_max = kk + BS;

                if (i_max > n) i_max = n;
                if (j_max > n) j_max = n;
                if (k_max > n) k_max = n;

                for (int i = ii; i < i_max; ++i) {
                    for (int j = jj; j < j_max; ++j) {

                        float sum0 = 0.0f;
                        float sum1 = 0.0f;
                        float sum2 = 0.0f;
                        float sum3 = 0.0f;

                        int k = kk;
                        int limit = (k_max - kk) & ~3;
                        limit += kk;

                        #pragma omp simd reduction(+:sum0,sum1,sum2,sum3)
                        for (; k < limit; k += 4) {
                            const float a0 = a[i * n + (k + 0)];
                            const float a1 = a[i * n + (k + 1)];
                            const float a2 = a[i * n + (k + 2)];
                            const float a3 = a[i * n + (k + 3)];

                            const std::size_t row0 = static_cast<std::size_t>(k + 0) * n;
                            const std::size_t row1 = static_cast<std::size_t>(k + 1) * n;
                            const std::size_t row2 = static_cast<std::size_t>(k + 2) * n;
                            const std::size_t row3 = static_cast<std::size_t>(k + 3) * n;

                            const float b0 = b[row0 + j];
                            const float b1 = b[row1 + j];
                            const float b2 = b[row2 + j];
                            const float b3 = b[row3 + j];

                            sum0 += a0 * b0;
                            sum1 += a1 * b1;
                            sum2 += a2 * b2;
                            sum3 += a3 * b3;
                        }

                        float sum = sum0 + sum1 + sum2 + sum3;

                        for (; k < k_max; ++k) {
                            sum += a[i * n + k] * b[static_cast<std::size_t>(k) * n + j];
                        }

                        c[static_cast<std::size_t>(i) * n + j] += sum;
                    }
                }
            }
        }
    }

    return c;
}
