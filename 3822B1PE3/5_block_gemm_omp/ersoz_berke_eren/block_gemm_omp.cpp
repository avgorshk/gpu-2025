#include "block_gemm_omp.h"

#include <omp.h>
#include <algorithm>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    if (n <= 0) {
        return {};
    }

    const size_t matrixSize = static_cast<size_t>(n) * n;
    std::vector<float> c(matrixSize, 0.0f);

    const int blockSize = 32;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int rowBlock = 0; rowBlock < n; rowBlock += blockSize) {
        for (int colBlock = 0; colBlock < n; colBlock += blockSize) {
            
            for (int kBlock = 0; kBlock < n; kBlock += blockSize) {
                int rowEnd = std::min(rowBlock + blockSize, n);
                int colEnd = std::min(colBlock + blockSize, n);
                int kEnd = std::min(kBlock + blockSize, n);

                for (int i = rowBlock; i < rowEnd; ++i) {
                    for (int k = kBlock; k < kEnd; ++k) {
                        float aVal = a[i * n + k];
                        
                        // Loop unrolling by 4
                        int j = colBlock;
                        for (; j <= colEnd - 4; j += 4) {
                            c[i * n + j]     += aVal * b[k * n + j];
                            c[i * n + j + 1] += aVal * b[k * n + j + 1];
                            c[i * n + j + 2] += aVal * b[k * n + j + 2];
                            c[i * n + j + 3] += aVal * b[k * n + j + 3];
                        }
                        
                        // Handle remaining elements
                        for (; j < colEnd; ++j) {
                            c[i * n + j] += aVal * b[k * n + j];
                        }
                    }
                }
            }
        }
    }

    return c;
}
