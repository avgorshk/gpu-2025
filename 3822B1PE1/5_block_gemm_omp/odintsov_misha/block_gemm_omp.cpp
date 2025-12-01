#include "block_gemm_omp.h"

#include <vector>
#include <omp.h>

std::vector<float> BlockGemmOMP(const std::vector<float>& a, const std::vector<float>& b, int n)
{
    const int block_size = 32; 

    std::vector<float> c(n * n, 0.0f);

#pragma omp parallel for collapse(2)
    for (int i_block = 0; i_block < n; i_block += block_size) {
        for (int j_block = 0; j_block < n; j_block += block_size) {


            for (int i = i_block; i < i_block + block_size; ++i) {
                for (int j = j_block; j < j_block + block_size; ++j) {
                    float sum = 0.0f;

                    for (int k_block = 0; k_block < n; k_block += block_size) {
                    
                        #pragma omp simd reduction(+:sum)  
                        for (int k = k_block; k < k_block + block_size; ++k) {
                            sum += a[i * n + k] * b[k * n + j];
                        }
                    }

                    #pragma omp atomic
                    c[i * n + j] += sum;
                }
            }
        }
    }




    return c;
}

