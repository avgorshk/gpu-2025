#include "block_gemm_omp.h"
#include <omp.h>
#include <algorithm> // std::min

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    const size_t size = static_cast<size_t>(n) * n;
    std::vector<float> c(size, 0.0f);

    // Vérification simple des tailles
    if (n <= 0 || a.size() != size || b.size() != size) {
        return c;
    }

    // Taille de bloc : compromis simple pour CPU moderne
    const int BS = 64;

    // Parcours par blocs (I, J, K)
    // On parallélise sur les blocs (I, J) -> régions distinctes de C, donc pas de data race
#pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < n; ii += BS) {
        for (int jj = 0; jj < n; jj += BS) {

            for (int kk = 0; kk < n; kk += BS) {
                int i_max = std::min(ii + BS, n);
                int j_max = std::min(jj + BS, n);
                int k_max = std::min(kk + BS, n);

                // Petit GEMM dans chaque bloc
                for (int i = ii; i < i_max; ++i) {
                    int c_row_offset = i * n;
                    int a_row_offset = i * n;

                    for (int k = kk; k < k_max; ++k) {
                        float aik = a[a_row_offset + k];  // A[i, k]
                        int b_row_offset = k * n;         // début de la ligne k de B

#pragma omp simd
                        for (int j = jj; j < j_max; ++j) {
                            c[c_row_offset + j] += aik * b[b_row_offset + j];
                        }
                    }
                }
            }
        }
    }

    return c;
}
