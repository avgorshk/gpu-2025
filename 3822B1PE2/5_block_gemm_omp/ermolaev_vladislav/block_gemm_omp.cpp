#include "block_gemm_omp.h"
#include <vector>
#include <omp.h>

#define MATRIX_BLOCK_SIZE 64
#define TRANSPOSE_BLOCK 16

void TransposeMatrix(const std::vector<float>& input, std::vector<float>& output, int n) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; i += TRANSPOSE_BLOCK) {
        for (int j = 0; j < n; j += TRANSPOSE_BLOCK) {
            for (int ii = i; ii < std::min(i + TRANSPOSE_BLOCK, n); ++ii) {
                for (int jj = j; jj < std::min(j + TRANSPOSE_BLOCK, n); ++jj) {
                    output[jj * n + ii] = input[ii * n + jj];
                }
            }
        }
    }
}

void MultiplyBlock(const std::vector<float>& a, const std::vector<float>& bt, 
                   std::vector<float>& c, int n, int bi, int bj, int num_blocks_k) {
    int block_size_i = std::min(MATRIX_BLOCK_SIZE, n - bi * MATRIX_BLOCK_SIZE);
    int block_size_j = std::min(MATRIX_BLOCK_SIZE, n - bj * MATRIX_BLOCK_SIZE);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < block_size_i; ++i) {
        for (int j = 0; j < block_size_j; ++j) {
            float sum = 0.0f;

            #pragma unroll 4
            for (int bk = 0; bk < num_blocks_k; ++bk) {
                int block_size_k = std::min(MATRIX_BLOCK_SIZE, n - bk * MATRIX_BLOCK_SIZE);
                int ak = bk * MATRIX_BLOCK_SIZE;
                int bj_row = bj * MATRIX_BLOCK_SIZE + j;
                
                
                #pragma unroll 4
                for (int k = 0; k < block_size_k; ++k) {
                    int ai = bi * MATRIX_BLOCK_SIZE + i;
                    sum += a[ai * n + (ak + k)] * bt[bj_row * n + (ak + k)];
                }
            }
            
            int ci = bi * MATRIX_BLOCK_SIZE + i;
            int cj = bj * MATRIX_BLOCK_SIZE + j;
            c[ci * n + cj] = sum;
        }
    }
}

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(n * n, 0.0f);
    std::vector<float> bt(n * n);
    TransposeMatrix(b, bt, n);
    
    int num_blocks = (n + MATRIX_BLOCK_SIZE - 1) / MATRIX_BLOCK_SIZE;
    int num_blocks_k = num_blocks;
    
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int bi = 0; bi < num_blocks; ++bi) {
        for (int bj = 0; bj < num_blocks; ++bj) {
            MultiplyBlock(a, bt, c, n, bi, bj, num_blocks_k);
        }
    }
    
    return c;
}