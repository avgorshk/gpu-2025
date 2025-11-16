#include "block_gemm_omp.h"

#include <vector>
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <omp.h>
#ifndef BLOCK_SZ
#define BLOCK_SZ 32
#endif

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {
    if (n <= 0) return {};
    const size_t nn = static_cast<size_t>(n) * static_cast<size_t>(n);
    if (a.size() != nn || b.size() != nn) return {};
    std::vector<float> c(nn, 0.0f);

    std::vector<float> bT(nn);
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            bT[j * n + i] = b[i * n + j];
        }
    }

    const int BS = BLOCK_SZ;

#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int ii = 0; ii < n; ii += BS) {
        for (int jj = 0; jj < n; jj += BS) {

            int i_end = std::min(ii + BS, n);
            int j_end = std::min(jj + BS, n);

            for (int kk = 0; kk < n; kk += BS) {
                int k_end = std::min(kk + BS, n);

                for (int i = ii; i < i_end; ++i) {
                    const float* a_row = &a[i * n];
                    float* c_row = &c[i * n];

                    for (int j = jj; j < j_end; ++j) {
                        const float* bT_row = &bT[j * n];
                        float sum = c_row[j];

                        int k = kk;
                        for (; k <= k_end - 4; k += 4) {
                            sum += a_row[k + 0] * bT_row[k + 0];
                            sum += a_row[k + 1] * bT_row[k + 1];
                            sum += a_row[k + 2] * bT_row[k + 2];
                            sum += a_row[k + 3] * bT_row[k + 3];
                        }
                        for (; k < k_end; ++k) {
                            sum += a_row[k] * bT_row[k];
                        }

                        c_row[j] = sum;
                    }
                }
            }
        }
    }

    return c;
}
