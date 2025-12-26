#include "block_gemm_omp.h"
#include <vector>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    const int BLOCK_SIZE = 32;
    std::vector<float> c(n * n, 0.0f);
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i += BLOCK_SIZE) {
        for (int j = 0; j < n; j += BLOCK_SIZE) {
            for (int k = 0; k < n; k += BLOCK_SIZE) {
                int i_end = std::min(i + BLOCK_SIZE, n);
                int j_end = std::min(j + BLOCK_SIZE, n);
                int k_end = std::min(k + BLOCK_SIZE, n);
                
                for (int ii = i; ii < i_end; ++ii) {
                    for (int jj = j; jj < j_end; ++jj) {
                        float sum = c[ii * n + jj];
                        for (int kk = k; kk < k_end; ++kk) {
                            sum += a[ii * n + kk] * b[kk * n + jj];
                        }
                        c[ii * n + jj] = sum;
                    }
                }
            }
        }
    }
    
    return c;
}