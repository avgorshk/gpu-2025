#include "naive_gemm_omp.h"
#include <vector>
#include <omp.h>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    if (n <= 0) {
        return {};
    }
    
    std::vector<float> c(n * n, 0.0f);
    
    std::vector<float> b_transposed(n * n);
    
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            b_transposed[j * n + i] = b[i * n + j];
        }
    }
    
    const float* __restrict a_ptr = a.data();
    const float* __restrict bt_ptr = b_transposed.data();
    float* __restrict c_ptr = c.data();
    
    constexpr int UNROLL_FACTOR = 4;
    
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        const int i_offset = i * n;
        
        for (int j = 0; j < n; ++j) {
            const int j_offset = j * n;
            float sum = 0.0f;
            int k = 0;
            
            for (; k <= n - UNROLL_FACTOR; k += UNROLL_FACTOR) {
                sum += a_ptr[i_offset + k] * bt_ptr[j_offset + k] +
                       a_ptr[i_offset + k + 1] * bt_ptr[j_offset + k + 1] +
                       a_ptr[i_offset + k + 2] * bt_ptr[j_offset + k + 2] +
                       a_ptr[i_offset + k + 3] * bt_ptr[j_offset + k + 3];
            }
            
            // Handle remaining elements
            for (int k_rem = k; k_rem < n; ++k_rem) {
                sum += a_ptr[i_offset + k_rem] * bt_ptr[j_offset + k_rem];
            }
            
            c_ptr[i_offset + j] = sum;
        }
    }
    
    return c;
}

