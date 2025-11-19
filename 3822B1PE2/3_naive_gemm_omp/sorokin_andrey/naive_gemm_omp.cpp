#include "naive_gemm_omp.h"
#include <cstring>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int dim) {
    std::vector<float> c(dim * dim, 0.0f);
    std::vector<float> b_t(dim * dim);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            b_t[j * dim + i] = b[i * dim + j];
        }
    }
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            float total = 0.0f;
            int idx = 0;
            for (; idx <= dim - 4; idx += 4) {
                total += a[i * dim + idx] * b_t[j * dim + idx] +
                         a[i * dim + idx + 1] * b_t[j * dim + idx + 1] +
                         a[i * dim + idx + 2] * b_t[j * dim + idx + 2] +
                         a[i * dim + idx + 3] * b_t[j * dim + idx + 3];
            }
            for (; idx < dim; idx++) {
                total += a[i * dim + idx] * b_t[j * dim + idx];
            }
            
            c[i * dim + j] = total;
        }
    }

    return c;
}