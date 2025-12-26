#include "block_gemm_omp.h"

#include <omp.h>

std::vector<float> BlockGemmOMP(const std::vector<float> &a,
                                const std::vector<float> &b,
                                int n) {
    std::vector<float> res(n * n, 0);
    int size_block_x = 4;
    int size_block_y = 8;
    
    #pragma omp parallel for
    for (int ib = 0; ib < n; ib += size_block_y) {
        for (int kb = 0; kb < n; kb += size_block_x) {
            for (int jb = 0; jb < n; jb += size_block_y) {
                int iEnd = std::min(n, ib + size_block_y);
                int jEnd = std::min(n, jb + size_block_y);
                int kEnd = std::min(n, kb + size_block_x);
        
                for (int i = ib; i < iEnd; i++) {
                    for (int k = kb; k < kEnd; k++) {
                        #pragma omp simd
                        for (int j = jb; j < jEnd; j++) {
                            res[i * n + j] += a[i * n + k] * b[k * n + j];
                        }
                    }
                }
            }
        }
    }
    
    return res;
}
