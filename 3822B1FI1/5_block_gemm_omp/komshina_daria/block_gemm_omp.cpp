#include "block_gemm_omp.h"
#include <omp.h>
#include <algorithm>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {
    std::vector<float> c(n * n, 0.0f);

    int blockSize = 64;
    if (blockSize > n) blockSize = n;

#pragma omp parallel for schedule(dynamic)
    for (int ii = 0; ii < n; ii += blockSize) {
        for (int kk = 0; kk < n; kk += blockSize) {
            for (int jj = 0; jj < n; jj += blockSize) {

                int i_max = std::min(ii + blockSize, n);
                int k_max = std::min(kk + blockSize, n);
                int j_max = std::min(jj + blockSize, n);

                for (int i = ii; i < i_max; ++i) {
                    for (int k = kk; k < k_max; ++k) {
                        float a_ik = a[i * n + k];

                        int j = jj;
                        for (; j + 3 < j_max; j += 4) {
                            c[i * n + j] += a_ik * b[k * n + j];
                            c[i * n + j + 1] += a_ik * b[k * n + j + 1];
                            c[i * n + j + 2] += a_ik * b[k * n + j + 2];
                            c[i * n + j + 3] += a_ik * b[k * n + j + 3];
                        }
                        for (; j < j_max; ++j) {
                            c[i * n + j] += a_ik * b[k * n + j];
                        }
                    }
                }
            }
        }
    }

    return c;
}