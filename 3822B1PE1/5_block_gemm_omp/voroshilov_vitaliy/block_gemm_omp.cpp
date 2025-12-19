#include "block_gemm_omp.h"

#include <vector>
#include <omp.h>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(n * n, 0.0f);

    const int block_size = 16;

    #pragma omp parallel for
    for (int I = 0; I < n; I += block_size) 
        for (int J = 0; J < n; J += block_size) 
            for (int K = 0; K < n; K += block_size) 

                for (int i = I; i < I + block_size; i++) 
                    for (int k = K; k < K + block_size; k++) 
                        for (int j = J; j < J + block_size; j++) 
                            c[i * n + j] += a[i * n + k] * b[k * n + j];

    return c;
}