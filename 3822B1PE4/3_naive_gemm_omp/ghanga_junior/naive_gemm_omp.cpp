#include "naive_gemm_omp.h"
#include <omp.h>
#include <vector>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n)
{
    // Alloue la matrice résultat C = A * B
    std::vector<float> c(static_cast<size_t>(n) * n, 0.0f);

    // Cas sécuritaire si tailles invalides
    if (n <= 0 || a.size() != c.size() || b.size() != c.size()) {
        return c;
    }

    // GEMM naïf : C[i,j] = somme_k (A[i,k] * B[k,j])
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {

            float sum = 0.0f;

            for (int k = 0; k < n; ++k) {
                sum += a[i * n + k] * b[k * n + j];
            }

            c[i * n + j] = sum;
        }
    }

    return c;
}
