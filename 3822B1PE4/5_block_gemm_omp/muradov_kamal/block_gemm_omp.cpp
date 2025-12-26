#include "block_gemm_omp.h"

#include <algorithm>
#include <vector>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(static_cast<size_t>(n) * n, 0.0f);
    int block = (n >= 64) ? 64 : n;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int I = 0; I < n; I += block) {
        for (int J = 0; J < n; J += block) {
            for (int K = 0; K < n; K += block) {
                int i_end = std::min(I + block, n);
                int j_end = std::min(J + block, n);
                int k_end = std::min(K + block, n);
                for (int i = I; i < i_end; ++i) {
                    int row = i * n;
                    for (int k = K; k < k_end; ++k) {
                        float a_ik = a[row + k];
                        int b_row = k * n;
                        for (int j = J; j < j_end; ++j) {
                            c[row + j] += a_ik * b[b_row + j];
                        }
                    }
                }
            }
        }
    }

    return c;
}
