#include "block_gemm_omp.h"
#include <omp.h>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n)
{
    std::vector<float> c(n * n, 0.0f);

    const int BS = 64;

    std::vector<float> bT(n * n);
    #pragma omp parallel for
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            bT[j*n + i] = b[i*n + j];

    #pragma omp parallel for collapse(2)
    for (int ii = 0; ii < n; ii += BS) {
        for (int jj = 0; jj < n; jj += BS) {

            for (int kk = 0; kk < n; kk += BS) {

                int i_end = ii + BS;
                int j_end = jj + BS;
                int k_end = kk + BS;

                for (int i = ii; i < i_end; i++) {
                    float* crow = &c[i * n];
                    const float* arow = &a[i * n];

                    for (int j = jj; j < j_end; j++) {
                        const float* brow = &bT[j * n];

                        float sum = crow[j];

                        int k = kk;

                        for (; k <= k_end - 4; k += 4) {
                            sum += arow[k]     * brow[k];
                            sum += arow[k + 1] * brow[k + 1];
                            sum += arow[k + 2] * brow[k + 2];
                            sum += arow[k + 3] * brow[k + 3];
                        }

                        for (; k < k_end; k++)
                            sum += arow[k] * brow[k];

                        crow[j] = sum;
                    }
                }

            }
        }
    }

    return c;
}
