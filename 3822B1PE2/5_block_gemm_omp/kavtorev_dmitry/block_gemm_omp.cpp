#include "block_gemm_omp.h"
#include <omp.h>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    const int BLOCK_SIZE = 32;
    std::vector<float> c(n * n, 0.0f);
    
    #pragma omp parallel for collapse(2) schedule(static)
    for (int bi = 0; bi < n; bi += BLOCK_SIZE) {
        for (int bj = 0; bj < n; bj += BLOCK_SIZE) {
            for (int bk = 0; bk < n; bk += BLOCK_SIZE) {
                for (int i = bi; i < bi + BLOCK_SIZE && i < n; ++i) {
                    for (int j = bj; j < bj + BLOCK_SIZE && j < n; ++j) {
                        float sum = c[i * n + j];
                        for (int k = bk; k < bk + BLOCK_SIZE && k < n; ++k) {
                            sum += a[i * n + k] * b[k * n + j];
                        }
                        c[i * n + j] = sum;
                    }
                }
            }
        }
    }
    
    return c;
}

