//
// Created by korablev-vm on 29.09.2025.
//

#include "block_gemm_omp.h"

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector answer(n * n, 0.0f);
    std::vector<float> bt(n * n);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            bt[j * n + i] = b[i * n + j];
        }
    }

    constexpr int block_sz = 16;

    if (n % block_sz == 0) {
#pragma omp parallel for collapse(2) schedule(dynamic)
        for (int ii = 0; ii < n; ii += block_sz) {
            for (int jj = 0; jj < n; jj += block_sz) {
                for (int kk = 0; kk < n; kk += block_sz) {
                    for (int i = 0; i < block_sz; i++) {
                        for (int j = 0; j < block_sz; j++) {
                            float sum = 0.0f;
                            for (int k = 0; k < block_sz; k++) {
                                sum += a[(ii + i) * n + (kk + k)] *
                                       bt[(jj + j) * n + (kk + k)];
                            }
                            answer[(ii + i) * n + (jj + j)] += sum;
                        }
                    }
                }
            }
        }
    } else {
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0.0f;
                for (int k = 0; k < n; k++) {
                    sum += a[i * n + k] * bt[j * n + k];
                }
                answer[i * n + j] = sum;
            }
        }
    }
    return answer;
}
