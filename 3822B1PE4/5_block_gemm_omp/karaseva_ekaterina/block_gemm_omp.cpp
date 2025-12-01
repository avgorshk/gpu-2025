#include "block_gemm_omp.h"
#include <vector>
#include <algorithm>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(n * n, 0.0f);
    int block_size = std::min(64, n);
    
    #pragma omp parallel for collapse(2)
    for (int I = 0; I < n; I += block_size) {
        for (int J = 0; J < n; J += block_size) {
            for (int K = 0; K < n; K += block_size) {
                int i_end = std::min(I + block_size, n);
                int j_end = std::min(J + block_size, n);
                int k_end = std::min(K + block_size, n);
                
                for (int i = I; i < i_end; ++i) {
                    for (int k = K; k < k_end; ++k) {
                        float a_ik = a[i * n + k];
                        for (int j = J; j < j_end; ++j) {
                            c[i * n + j] += a_ik * b[k * n + j];
                        }
                    }
                }
            }
        }
    }
    return c;
}