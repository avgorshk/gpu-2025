#include "block_gemm_omp.h"
#include <vector>
#include <omp.h>
#include <algorithm>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(n * n, 0.0f);

    const int block_size = 64;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int bi = 0; bi < n; bi += block_size) {
        for (int bj = 0; bj < n; bj += block_size) {
            for (int bk = 0; bk < n; bk += block_size) {

                for (int i = bi; i < std::min(bi + block_size, n); ++i) {
                    int i_n = i * n;
                    for (int k = bk; k < std::min(bk + block_size, n); ++k) {
                        float a_ik = a[i_n + k];
                        int k_n = k * n;

                        #pragma omp simd
                        for (int j = bj; j < std::min(bj + block_size, n); ++j) {
                            c[i_n + j] += a_ik * b[k_n + j];
                        }
                    }
                }

            }
        }
    }

    return c;
}
