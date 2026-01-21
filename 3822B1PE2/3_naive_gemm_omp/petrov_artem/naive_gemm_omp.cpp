#include "naive_gemm_omp.h"
#include <omp.h>
#include <vector>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,const std::vector<float>& b,int n) {
    std::vector<float> c(n * n, 0.0f);
    
    #pragma omp parallel
    {
        #pragma omp for schedule(static) collapse(2)
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                float sum = 0.0f;
                
                const float* a_ptr = &a[i * n];
                const float* b_ptr = &b[j];
                
                int k = 0;
                for (; k + 3 < n; k += 4) {
                    sum += a_ptr[k] * b_ptr[k * n] + a_ptr[k + 1] * b_ptr[(k + 1) * n] + a_ptr[k + 2] * b_ptr[(k + 2) * n] + a_ptr[k + 3] * b_ptr[(k + 3) * n];
                }
                
                for (; k < n; ++k) {
                    sum += a_ptr[k] * b_ptr[k * n];
                }
                
                c[i * n + j] = sum;
            }
        }
    }
    
    return c;
}
