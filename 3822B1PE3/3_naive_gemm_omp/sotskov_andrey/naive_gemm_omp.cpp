#include "naive_gemm_omp.h"
#include <vector>
#include <algorithm>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b, 
                                int n) {
    if (n <= 0) return {};
    if (a.size() != size_t(n * n) || b.size() != size_t(n * n)) {
        return {};
    }
    
    std::vector<float> result(n * n, 0.0f);
    const int blockSize = 64;
    
    #pragma omp parallel for schedule(static)
    for (int block_i = 0; block_i < n; block_i += blockSize) {
        for (int block_k = 0; block_k < n; block_k += blockSize) {
            for (int block_j = 0; block_j < n; block_j += blockSize) {
                int block_i_end = std::min(block_i + blockSize, n);
                int block_j_end = std::min(block_j + blockSize, n);
                int block_k_end = std::min(block_k + blockSize, n);
                
                for (int i = block_i; i < block_i_end; ++i) {
                    for (int k = block_k; k < block_k_end; ++k) {
                        float a_element = a[i * n + k];
                        
                        int j = block_j;

                        for (; j <= block_j_end - 4; j += 4) {
                            result[i * n + j] += a_element * b[k * n + j];
                            result[i * n + j + 1] += a_element * b[k * n + j + 1];
                            result[i * n + j + 2] += a_element * b[k * n + j + 2];
                            result[i * n + j + 3] += a_element * b[k * n + j + 3];
                        }

                        for (; j < block_j_end; ++j) {
                            result[i * n + j] += a_element * b[k * n + j];
                        }
                    }
                }
            }
        }
    }
    return result;
}