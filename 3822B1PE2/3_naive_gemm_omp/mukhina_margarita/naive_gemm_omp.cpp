#include "naive_gemm_omp.h"
#include <omp.h>
#include <vector>
#include <algorithm>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                               const std::vector<float>& b,
                               int n) {
    std::vector<float> c(n * n, 0.0f);
    
    std::vector<float> b_t(n * n);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            b_t[j * n + i] = b[i * n + j];
        }
    }
    
    const int blockSize = 64;
    
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; i += blockSize) {
        for (int j = 0; j < n; j += blockSize) {
            int i_end = std::min(i + blockSize, n);
            int j_end = std::min(j + blockSize, n);
            
            for (int k = 0; k < n; ++k) {
                for (int bi = i; bi < i_end; ++bi) {
                    float a_val = a[bi * n + k];
                    float* c_row = &c[bi * n + j]; 
                    const float* b_t_col = &b_t[j * n + k]; 
                    
                    int bj = j;
                    for (; bj <= j_end - 4; bj += 4) {
                        c_row[bj - j] += a_val * b_t_col[bj - j];
                        c_row[bj - j + 1] += a_val * b_t_col[bj - j + 1];
                        c_row[bj - j + 2] += a_val * b_t_col[bj - j + 2];
                        c_row[bj - j + 3] += a_val * b_t_col[bj - j + 3];
                    }
                    
                    for (; bj < j_end; ++bj) {
                        c_row[bj - j] += a_val * b_t_col[bj - j];
                    }
                }
            }
        }
    }
    
    return c;
}