#include <vector>
#include <omp.h>
#include <algorithm>

constexpr int TILE_SIZE = 32;

void MultiplyBlock(const std::vector<float>& A, const std::vector<float>& B,
    std::vector<float>& C, int dim,
    int rowOffset, int colOffset, int kOffset, int tileSize) {

    int rowLimit = std::min(rowOffset + tileSize, dim);
    int colLimit = std::min(colOffset + tileSize, dim);
    int kLimit = std::min(kOffset + tileSize, dim);

    for (int i = rowOffset; i < rowLimit; ++i) {
        for (int k = kOffset; k < kLimit; ++k) {
            float a_val = A[i * dim + k];
#pragma omp simd
            for (int j = colOffset; j < colLimit; ++j) {
                C[i * dim + j] += a_val * B[k * dim + j];
            }
        }
    }
}

std::vector<float> BlockGemmOMP(const std::vector<float>& A,
    const std::vector<float>& B,
    int N) {
    std::vector<float> result(N * N, 0.0f);

#pragma omp parallel for collapse(2) schedule(static)
    for (int row = 0; row < N; row += TILE_SIZE) {
        for (int col = 0; col < N; col += TILE_SIZE) {
            for (int k = 0; k < N; k += TILE_SIZE) {
                MultiplyBlock(A, B, result, N, row, col, k, TILE_SIZE);
            }
        }
    }

    return result;
}