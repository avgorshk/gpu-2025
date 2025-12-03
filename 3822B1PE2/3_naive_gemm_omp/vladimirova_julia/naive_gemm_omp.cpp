#include "naive_gemm_omp.h"
#include <omp.h>
#include <vector>




std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {

    std::vector<float> c(n * n, 0.0f);  
    
    


    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        float* local_c = &c[i * n];
        const float* local_a = &a[i * n];
        
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += local_a[k] * b[k * n + j];
            }
            local_c[j] = sum;
        }
    }
    return c;
    // Place your implementation here
}