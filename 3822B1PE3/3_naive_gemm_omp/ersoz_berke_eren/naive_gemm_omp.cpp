#include "naive_gemm_omp.h"

#include <omp.h>
#include <algorithm>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    if (n <= 0) {
        return {};
    }
    
    const size_t matrixSize = static_cast<size_t>(n) * n;
    if (a.size() != matrixSize || b.size() != matrixSize) {
        return {};
    }

    std::vector<float> c(matrixSize, 0.0f);

    // Transpose B for better cache locality
    std::vector<float> bTransposed(matrixSize);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            bTransposed[j * n + i] = b[i * n + j];
        }
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            
            // Loop unrolling by 4
            int k = 0;
            for (; k <= n - 4; k += 4) {
                sum += a[i * n + k] * bTransposed[j * n + k];
                sum += a[i * n + k + 1] * bTransposed[j * n + k + 1];
                sum += a[i * n + k + 2] * bTransposed[j * n + k + 2];
                sum += a[i * n + k + 3] * bTransposed[j * n + k + 3];
            }
            
            // Handle remaining elements
            for (; k < n; ++k) {
                sum += a[i * n + k] * bTransposed[j * n + k];
            }
            
            c[i * n + j] = sum;
        }
    }

    return c;
}
