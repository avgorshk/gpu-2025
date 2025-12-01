#include "block_gemm_omp.h"

#include <vector>
#include <omp.h>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {

    std::vector<float> c(n * n, 0.0f);

    constexpr int block_size = 64;

#pragma omp parallel for collapse(2)
    for (int i0 = 0; i0 < n; i0 += block_size) {
        for (int j0 = 0; j0 < n; j0 += block_size) {
            for (int k0 = 0; k0 < n; k0 += block_size) {

                int i1 = std::min(i0 + block_size, n);
                int j1 = std::min(j0 + block_size, n);
                int k1 = std::min(k0 + block_size, n);

                for (int i = i0; i < i1; i++) {
                    for (int j = j0; j < j1; j++) {
                        float sum = c[i * n + j];
                        for (int k = k0; k < k1; k++) {
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