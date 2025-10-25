#include "block_gemm_omp.h"
#include <vector>
#include <algorithm>
#include <cstddef>   
#include <omp.h>     

constexpr int BLOCK_TILE_SIZE = 64;
constexpr int UNROLL_J = 4;

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {
    if (n <= 0) {
        return {};
    }

    size_t total_size = static_cast<size_t>(n) * n;
    if (a.size() != total_size || b.size() != total_size) {
        return {};
    }

    std::vector<float> c(total_size, 0.0f);
    int num_blocks = (n + BLOCK_TILE_SIZE - 1) / BLOCK_TILE_SIZE;

#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int block_i = 0; block_i < num_blocks; ++block_i) {
        for (int block_j = 0; block_j < num_blocks; ++block_j) {
            int i_start = block_i * BLOCK_TILE_SIZE;
            int i_end = std::min(i_start + BLOCK_TILE_SIZE, n);
            int j_start = block_j * BLOCK_TILE_SIZE;
            int j_end = std::min(j_start + BLOCK_TILE_SIZE, n);
            for (int block_k = 0; block_k < num_blocks; ++block_k) {
                int k_start = block_k * BLOCK_TILE_SIZE;
                int k_end = std::min(k_start + BLOCK_TILE_SIZE, n);
                for (int i = i_start; i < i_end; ++i) {
                    float* c_ptr = &c[static_cast<size_t>(i) * n];
                    const float* a_ptr = &a[static_cast<size_t>(i) * n];
                    for (int k = k_start; k < k_end; ++k) {
                        float a_val = a_ptr[k];
                        int j = j_start;
                        for (; j <= j_end - UNROLL_J; j += UNROLL_J) {
                            c_ptr[j + 0] += a_val * b[static_cast<size_t>(k) * n + (j + 0)];
                            c_ptr[j + 1] += a_val * b[static_cast<size_t>(k) * n + (j + 1)];
                            c_ptr[j + 2] += a_val * b[static_cast<size_t>(k) * n + (j + 2)];
                            c_ptr[j + 3] += a_val * b[static_cast<size_t>(k) * n + (j + 3)];
                        }
                        for (; j < j_end; ++j) {
                            c_ptr[j] += a_val * b[static_cast<size_t>(k) * n + j];
                        }
                    }
                }
            }
        }
    }
    return c;
}
