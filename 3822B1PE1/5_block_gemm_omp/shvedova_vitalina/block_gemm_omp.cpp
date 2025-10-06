#include "block_gemm_omp.h"

#include <omp.h>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(n * n, 0.0f);
    constexpr int block_sz = 16;

    if (n % block_sz == 0) {
#pragma omp parallel for collapse(2) schedule(static)
        for (int ii = 0; ii < n; ii += block_sz) {
            for (int jj = 0; jj < n; jj += block_sz) {
                for (int kk = 0; kk < n; kk += block_sz) {
                    for (int i = 0; i < block_sz; i++) {
                        for (int k = 0; k < block_sz; k++) {
                            float aval = a[(ii + i) * n + (kk + k)];
                            float* crow = &c[(ii + i) * n + jj];
                            const float* brow = &b[(kk + k) * n + jj];
                            for (int j = 0; j < block_sz; j++) {
                                crow[j] += aval * brow[j];
                            }
                        }
                    }
                }
            }
        }
    } else {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n; k++) {
                float aval = a[i * n + k];
                const float* brow = &b[k * n];
                float* crow = &c[i * n];
                for (int j = 0; j < n; j++) {
                    crow[j] += aval * brow[j];
                }
            }
        }
    }
    return c;
}
