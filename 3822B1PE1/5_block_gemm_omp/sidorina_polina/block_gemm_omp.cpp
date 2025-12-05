#include "block_gemm_omp.h"
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <vector>

static void MultiplyBlock(const float* matrix_a,
                         const float* matrix_b,
                         float* matrix_c,
                         int block_row_start,
                         int block_col_start,
                         int block_size,
                         int matrix_size)
{
    int row_end = std::min(block_row_start + block_size, matrix_size);
    int col_end = std::min(block_col_start + block_size, matrix_size);
    
    for (int i = block_row_start; i < row_end; ++i)
    {
        for (int j = block_col_start; j < col_end; ++j)
        {
            float sum = 0.0f;
            
            for (int k = 0; k < matrix_size; ++k)
            {
                sum += matrix_a[i * matrix_size + k] * matrix_b[k * matrix_size + j];
            }
            
            matrix_c[i * matrix_size + j] = sum;
        }
    }
}

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n)
{
    constexpr int DEFAULT_BLOCK_SIZE = 64;
    int block_size = DEFAULT_BLOCK_SIZE;
    
    if (n < block_size)
    {
        block_size = n;
    }
    
    std::vector<float> c(n * n, 0.0f);
    
    #pragma omp parallel for collapse(2)
    for (int block_i = 0; block_i < n; block_i += block_size)
    {
       for (int block_j = 0; block_j < n; block_j += block_size)
       {
           MultiplyBlock(a.data(), b.data(), c.data(),
                        block_i, block_j, block_size, n);
       }
    }
    
    return c;
}