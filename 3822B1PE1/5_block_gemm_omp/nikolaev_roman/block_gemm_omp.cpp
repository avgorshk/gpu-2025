#include "block_gemm_omp.h"
#include <omp.h>
#include <cstring> 
#include <stdexcept>    

constexpr int BLOCK_SIZE = 64;

#if defined(_MSC_VER)
    #define RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
    #define RESTRICT __restrict__
#else
    #define RESTRICT
#endif

static void multiplyBlocks(const float* RESTRICT A_block,
                           const float* RESTRICT B_block,
                           float* RESTRICT C_block,
                           int n,
                           int block_size) {
    for (int i = 0; i < block_size; ++i) {
        for (int k = 0; k < block_size; ++k) {
            float a_val = A_block[i * n + k];
            for (int j = 0; j < block_size; ++j) {
                C_block[i * n + j] += a_val * B_block[k * n + j];
            }
        }
    }
}


std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    if (n <= 0) {
        throw std::invalid_argument("Matrix size n must be > 0");
    }
    if (n % BLOCK_SIZE != 0) {
        throw std::invalid_argument("Matrix size n must be a multiple of BLOCK_SIZE");
    }
    if ((int)a.size() != n * n || (int)b.size() != n * n) {
        throw std::invalid_argument("Input matrices must be of size n×n");
    }

    std::vector<float> c(n * n, 0.0f);

    const int num_blocks = n / BLOCK_SIZE;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int block_i = 0; block_i < num_blocks; ++block_i) {
        for (int block_j = 0; block_j < num_blocks; ++block_j) {
            float* C_block = &c[block_i * BLOCK_SIZE * n + block_j * BLOCK_SIZE];
            
            for (int block_k = 0; block_k < num_blocks; ++block_k) {
                const float* A_block = &a[block_i * BLOCK_SIZE * n + block_k * BLOCK_SIZE];
                const float* B_block = &b[block_k * BLOCK_SIZE * n + block_j * BLOCK_SIZE];
                
                multiplyBlocks(A_block, B_block, C_block, n, BLOCK_SIZE);
            }
        }
    }

    return c;
}