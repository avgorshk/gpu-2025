#include "block_gemm_omp.h"
#include <algorithm>
#include <stdexcept>
#include <omp.h>


std::vector<float> BlockGemmOMP(const std::vector<float>& a, const std::vector<float>& b, int n)
{
    const std::size_t expected = static_cast<std::size_t>(n) * n;
    if (a.size() != expected || b.size() != expected) {
        throw std::runtime_error("BlockGemmOMP: wrong matrix size");
    }

    std::vector<float> c(expected, 0.0f);

    const int BASE_BLOCK = 64;
    const int bs = std::min(BASE_BLOCK, n);

#pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < n; ii += bs) {
        for (int jj = 0; jj < n; jj += bs) {

            for (int kk = 0; kk < n; kk += bs) {

                const int i_max = std::min(ii + bs, n);
                const int j_max = std::min(jj + bs, n);
                const int k_max = std::min(kk + bs, n);

                for (int i = ii; i < i_max; ++i) {
                    const int rowA = i * n;
                    const int rowC = rowA;

                    for (int k = kk; k < k_max; ++k) {
                        const float a_ik = a[rowA + k];
                        const int rowB = k * n;

#pragma omp simd
                        for (int j = jj; j < j_max; ++j) {
                            c[rowC + j] += a_ik * b[rowB + j];
                        }
                    }
                }
            }
        }
    }

    return c;
}
