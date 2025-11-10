#include "block_gemm_omp.h"
#include <vector>
#include <omp.h>
#include <algorithm>

std::vector<float> BlockGemmOMP(const std::vector<float>& A,
                                const std::vector<float>& B,
                                int n) {
    std::vector<float> C(n * n, 0.0f);

    int block_size = 64;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int bi = 0; bi < n; bi += block_size) {
        for (int bj = 0; bj < n; bj += block_size) {
            for (int bk = 0; bk < n; bk += block_size) {

                int i_max = std::min(bi + block_size, n);
                int j_max = std::min(bj + block_size, n);
                int k_max = std::min(bk + block_size, n);

                for (int i = bi; i < i_max; ++i) {
                    for (int k = bk; k < k_max; ++k) {
                        float a_val = A[i * n + k];
                        for (int j = bj; j < j_max; ++j) {
                            C[i * n + j] += a_val * B[k * n + j];
                        }
                    }
                }
            }
        }
    }

    return C;
}
