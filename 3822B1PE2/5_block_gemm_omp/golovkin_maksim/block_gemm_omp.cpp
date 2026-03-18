#include "block_gemm_omp.h"
#include <omp.h>

static const int kBlockSize = 64;

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(n * n, 0.0f);

#pragma omp parallel for schedule(static) collapse(2)
    for (int ii = 0; ii < n; ii += kBlockSize) {
        for (int jj = 0; jj < n; jj += kBlockSize) {
            for (int kk = 0; kk < n; kk += kBlockSize) {
                int iEnd = std::min(ii + kBlockSize, n);
                int jEnd = std::min(jj + kBlockSize, n);
                int kEnd = std::min(kk + kBlockSize, n);
                for (int i = ii; i < iEnd; ++i) {
                    for (int k = kk; k < kEnd; ++k) {
                        float aik = a[i * n + k];
                        for (int j = jj; j < jEnd; ++j) {
                            c[i * n + j] += aik * b[k * n + j];
                        }
                    }
                }
            }
        }
    }

    return c;
}
