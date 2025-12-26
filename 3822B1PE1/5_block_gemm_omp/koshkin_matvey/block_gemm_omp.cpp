#include "block_gemm_omp.h"
#include <omp.h>
#include <vector>
#include <algorithm>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {
    if (n == 0) return {};

    std::vector<float> c(n * n, 0.0f);

    // Choose block size: must divide n, and n is power of 2
    // Ensure block_size <= n
    const int block_size = (n < 64) ? n : 64;
    const int block_count = n / block_size; // now always >= 1

#pragma omp parallel for collapse(2) schedule(static)
    for (int I = 0; I < block_count; ++I) {
        for (int J = 0; J < block_count; ++J) {
            for (int K = 0; K < block_count; ++K) {
                for (int i = 0; i < block_size; ++i) {
                    const int a_row = I * block_size + i;
                    for (int j = 0; j < block_size; ++j) {
                        const int b_col = J * block_size + j;
                        float sum = 0.0f;

                    #pragma omp simd reduction(+:sum)
                        for (int k = 0; k < block_size; ++k) {
                            const int a_col = K * block_size + k;
                            const int b_row = K * block_size + k;
                            sum += a[a_row * n + a_col] * b[b_row * n + b_col];
                        }

                        c[a_row * n + b_col] += sum;
                    }
                }
            }
        }
    }

    return c;
}