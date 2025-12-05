#include "naive_gemm_omp.h"
#include <omp.h>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    // Résultat C = A * B, taille n x n
    std::vector<float> c(static_cast<size_t>(n) * n, 0.0f);

    if (a.size() != c.size() || b.size() != c.size() || n <= 0) {
        // Cas "sécurité" si jamais les tailles ne correspondent pas
        return c;
    }

    // Naive GEMM avec optimisation simple :
    // - on parallélise la boucle sur les lignes i
    // - on garde j dans la boucle interne pour accès contigu dans B et C
    //
    // c[i,j] += a[i,k] * b[k,j]

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        int c_row_offset = i * n;

        for (int k = 0; k < n; ++k) {
            float aik = a[i * n + k];   // A[i,k]
            int b_row_offset = k * n;   // début de la ligne k de B

            // On vectorise la boucle j si possible (hint pour le compilateur)
#pragma omp simd
            for (int j = 0; j < n; ++j) {
                c[c_row_offset + j] += aik * b[b_row_offset + j];
            }
        }
    }

    return c;
}
