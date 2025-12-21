#include "block_gemm_omp.h"
#include <omp.h>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    const int BLOCK_SIZE = 64;
    std::vector<float> c(n * n, 0.0f);
    
    std::vector<float> b_transposed(n * n);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            b_transposed[j * n + i] = b[i * n + j];
        }
    }
    
    #pragma omp parallel
    {
        #pragma omp for collapse(2) schedule(dynamic, 1)
        for (int bi = 0; bi < n; bi += BLOCK_SIZE) {
            for (int bj = 0; bj < n; bj += BLOCK_SIZE) {
                float block_c[BLOCK_SIZE * BLOCK_SIZE] = {0.0f};
                
                for (int bk = 0; bk < n; bk += BLOCK_SIZE) {
                    for (int i = 0; i < BLOCK_SIZE && (bi + i) < n; ++i) {
                        for (int j = 0; j < BLOCK_SIZE && (bj + j) < n; ++j) {
                            float sum = block_c[i * BLOCK_SIZE + j];
                            const float* a_row = &a[(bi + i) * n + bk];
                            const float* b_col = &b_transposed[(bj + j) * n + bk];
                            
                            int k = 0;
                            for (; k < BLOCK_SIZE - 3 && (bk + k + 3) < n; k += 4) {
                                sum += a_row[k] * b_col[k];
                                sum += a_row[k + 1] * b_col[k + 1];
                                sum += a_row[k + 2] * b_col[k + 2];
                                sum += a_row[k + 3] * b_col[k + 3];
                            }
                            for (; k < BLOCK_SIZE && (bk + k) < n; ++k) {
                                sum += a_row[k] * b_col[k];
                            }
                            block_c[i * BLOCK_SIZE + j] = sum;
                        }
                    }
                }
                
                for (int i = 0; i < BLOCK_SIZE && (bi + i) < n; ++i) {
                    for (int j = 0; j < BLOCK_SIZE && (bj + j) < n; ++j) {
                        c[(bi + i) * n + (bj + j)] = block_c[i * BLOCK_SIZE + j];
                    }
                }
            }
        }
    }
    
    return c;
}

