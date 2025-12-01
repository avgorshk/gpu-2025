#include "block_gemm_omp.h"
#include <vector>  
#include <omp.h>     

std::vector<float> BlockGemmOMP(const std::vector<float>& a, const std::vector<float>& b, int n) {
    
    std::vector<float> c(n*n, 0.0);
    int num_threads = omp_get_max_threads();
    int block_size = 2;
    if (n <= 256) {
        block_size = (num_threads <= 4) ? 32 : 16;
    }
    else if (n <= 1024) {
        if (num_threads <= 4) block_size = 64;
        else if (num_threads <= 8) block_size = 48;
        else block_size = 32;
    }
    else {
        if (num_threads <= 4) block_size = 128;
        else if (num_threads <= 8) block_size = 96;
        else block_size = 64;
    }

    block_size = std::min(block_size, n);
        
#pragma omp parallel for collapse(2) schedule(dynamic, 1)
    for (int i_b = 0; i_b < n; i_b += block_size) {
        for (int j_b = 0; j_b < n; j_b += block_size) {

            int i_e = std::min(i_b + block_size, n);
            int j_e = std::min(j_b + block_size, n);

            for (int k_b = 0; k_b < n; k_b += block_size) {

                int k_e = std::min(k_b + block_size, n);

                for (int i = i_b; i < i_e; i++) {
                    const float* a_local = &a[i * n];
                    for (int k = k_b; k < k_e; k++) {
                        float a_ik = a_local[k];
                        const float* b_local = &b[k * n]; 
                        for (int j = j_b; j < j_e; j++) {
                            c[i * n + j] += a_ik * b_local[j];
                        }
                    }
                }
            }
        }
    }


    return c;
}
