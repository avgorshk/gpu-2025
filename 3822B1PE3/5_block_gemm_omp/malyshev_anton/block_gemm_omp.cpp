#include "block_gemm_omp.h"
#include <vector>
#include <omp.h>

#define MATRIX_BLOCK_SIZE 64
#define TRANSPOSE_BLOCK 16

void TransposeMatrix(const std::vector<float>& input, std::vector<float>& output, int size) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int blockRow = 0; blockRow < size; blockRow += TRANSPOSE_BLOCK) {
        for (int blockCol = 0; blockCol < size; blockCol += TRANSPOSE_BLOCK) {
            for (int row = blockRow; row < std::min(blockRow + TRANSPOSE_BLOCK, size); ++row) {
                for (int col = blockCol; col < std::min(blockCol + TRANSPOSE_BLOCK, size); ++col) {
                    output[col * size + row] = input[row * size + col];
                }
            }
        }
    }
}

void MultiplyBlock(const std::vector<float>& a, const std::vector<float>& bt, 
                   std::vector<float>& c, int size, int blockRow, int blockCol, int numBlocksK) {
    int rowSize = std::min(MATRIX_BLOCK_SIZE, size - blockRow * MATRIX_BLOCK_SIZE);
    int colSize = std::min(MATRIX_BLOCK_SIZE, size - blockCol * MATRIX_BLOCK_SIZE);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int row = 0; row < rowSize; ++row) {
        for (int col = 0; col < colSize; ++col) {
            float sum = 0.0f;

            #pragma unroll 4
            for (int blockK = 0; blockK < numBlocksK; ++blockK) {
                int kSize = std::min(MATRIX_BLOCK_SIZE, size - blockK * MATRIX_BLOCK_SIZE);
                int aKOffset = blockK * MATRIX_BLOCK_SIZE;
                int btRow = blockCol * MATRIX_BLOCK_SIZE + col;
                
                
                #pragma unroll 4
                for (int k = 0; k < kSize; ++k) {
                    int aRow = blockRow * MATRIX_BLOCK_SIZE + row;
                    sum += a[aRow * size + (aKOffset + k)] * bt[btRow * size + (aKOffset + k)];
                }
            }
            
            int resultRow = blockRow * MATRIX_BLOCK_SIZE + row;
            int resultCol = blockCol * MATRIX_BLOCK_SIZE + col;
            c[resultRow * size + resultCol] = sum;
        }
    }
}

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> result(n * n, 0.0f);
    std::vector<float> bTransposed(n * n);
    TransposeMatrix(b, bTransposed, n);
    
    int numBlocks = (n + MATRIX_BLOCK_SIZE - 1) / MATRIX_BLOCK_SIZE;
    int numBlocksK = numBlocks;
    
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int blockRow = 0; blockRow < numBlocks; ++blockRow) {
        for (int blockCol = 0; blockCol < numBlocks; ++blockCol) {
            MultiplyBlock(a, bTransposed, result, n, blockRow, blockCol, numBlocksK);
        }
    }
    
    return result;
}