#include "block_gemm_omp.h"
#include <vector>
#include <omp.h>

std::vector<float> BlockGemmOMP(const std::vector<float>& matA,
                                const std::vector<float>& matB,
                                int N) {
    std::vector<float> matC(N * N, 0.0f);

    int tileSize = 2;
    if (N <= 512) {
        tileSize = 64;
    } else if (N <= 2048) {
        tileSize = 128;
    } else {
        tileSize = 96;
    }
    tileSize = std::min(tileSize, N);
    omp_set_num_threads(4);
    tileSize = std::min(tileSize, N);

#pragma omp parallel for collapse(2) schedule(dynamic, 1)
    for (int blockI = 0; blockI < N; blockI += tileSize) {
        for (int blockJ = 0; blockJ < N; blockJ += tileSize) {
            int endI = std::min(blockI + tileSize, N);
            int endJ = std::min(blockJ + tileSize, N);

            for (int blockK = 0; blockK < N; blockK += tileSize) {
                int endK = std::min(blockK + tileSize, N);

                for (int row = blockI; row < endI; ++row) {
                    const float* aRowPtr = &matA[row * N];
                    for (int mid = blockK; mid < endK; ++mid) {
                        float aVal = aRowPtr[mid];
                        const float* bRowPtr = &matB[mid * N];
                        for (int col = blockJ; col < endJ; ++col) {
                            matC[row * N + col] += aVal * bRowPtr[col];
                        }
                    }
                }
            }
        }
    }

    return matC;
}

