#include "naive_gemm_omp.h"
#include <vector>
#include <omp.h>
#include <cstddef>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {

    if (n <= 0) {
        return {};
    }

    const size_t matrix_size = static_cast<size_t>(n) * n;
    std::vector<float> c(matrix_size, 0.0f);

    const float* a_data = a.data();
    const float* b_data = b.data();
    float* c_data = c.data();

    const int UNROLL_FACTOR = 4;
    std::vector<float> b_transposed(matrix_size);

#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            b_transposed[j * n + i] = b_data[i * n + j];
        }
    }

    const float* bt_data = b_transposed.data();

#pragma omp parallel for collapse(2) schedule(static)
    for (int i_block = 0; i_block < n; i_block += UNROLL_FACTOR) {
        for (int j_block = 0; j_block < n; j_block += UNROLL_FACTOR) {

            float c_reg[UNROLL_FACTOR][UNROLL_FACTOR] = { {0.0f} };

            for (int k = 0; k < n; k++) {
                for (int i_inner = 0; i_inner < UNROLL_FACTOR; i_inner++) {
                    int i = i_block + i_inner;
                    if (i >= n) continue;

                    float a_val = a_data[i * n + k];
                    for (int j_inner = 0; j_inner < UNROLL_FACTOR; j_inner++) {
                        int j = j_block + j_inner;
                        if (j >= n) continue;

                        c_reg[i_inner][j_inner] += a_val * bt_data[j * n + k];
                    }
                }
            }
            
            for (int i_inner = 0; i_inner < UNROLL_FACTOR; i_inner++) {
                int i = i_block + i_inner;
                if (i >= n) break;

                for (int j_inner = 0; j_inner < UNROLL_FACTOR; j_inner++) {
                    int j = j_block + j_inner;
                    if (j >= n) break;

                    c_data[i * n + j] = c_reg[i_inner][j_inner];
                }
            }
        }
    }

    return c;
}