
#include "block_gemm_omp.h"

#include <algorithm>
#include <omp.h>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {

    const int PREFERRED = 64;
    int bs = std::min(PREFERRED, n);

    std::vector<float> c((size_t)n * n, 0.0f);

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int ii = 0; ii < n; ii += bs) {
        for (int jj = 0; jj < n; jj += bs) {
            for (int kk = 0; kk < n; kk += bs) {
                int i_max = std::min(ii + bs, n);
                int j_max = std::min(jj + bs, n);
                int k_max = std::min(kk + bs, n);

                for (int i = ii; i < i_max; ++i) {
                    size_t a_row = (size_t)i * n;
                    size_t c_row = (size_t)i * n;

                    for (int k = kk; k < k_max; ++k) {
                        float a_ik = a[a_row + k];
                        size_t b_row = (size_t)k * n;
                        for (int j = jj; j < j_max; ++j) {
                            c[c_row + j] += a_ik * b[b_row + j];
                        }
                    }
                }
            }
        }
    }

    return c;
}
