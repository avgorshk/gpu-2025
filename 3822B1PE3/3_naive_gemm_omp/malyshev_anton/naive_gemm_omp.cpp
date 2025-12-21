#include "naive_gemm_omp.h"
#include <vector>
#include <omp.h>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> result(n * n, 0.0f);
    
    #pragma omp parallel for
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            for (int k = 0; k < n; k++) {
                result[row * n + col] += a[row * n + k] * b[k * n + col];
            }
        }
    }
    
    return result;
}