#include "naive_gemm_omp.h"
#include <omp.h>
#include <vector>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    const int BLOCK_SIZE = 64;
    std::vector<float> c(n * n, 0.0f);
    
    #pragma omp parallel for collapse(2)
    for (int i0 = 0; i0 < n; i0 += BLOCK_SIZE) {
        for (int j0 = 0; j0 < n; j0 += BLOCK_SIZE) {
            for (int k0 = 0; k0 < n; k0 += BLOCK_SIZE) {
                for (int i = i0; i < std::min(i0+BLOCK_SIZE, n); ++i) {
                    for (int j = j0; j < std::min(j0+BLOCK_SIZE, n); ++j) {
                        float sum = 0.0f;
                        #pragma omp simd reduction(+:sum)
                        for (int k = k0; k < std::min(k0+BLOCK_SIZE, n); ++k) {
                            sum += a[i * n + k] * b[k * n + j];
                        }
                        c[i * n + j] += sum;
                    }
                }
            }
        }
    }
    return c;
}