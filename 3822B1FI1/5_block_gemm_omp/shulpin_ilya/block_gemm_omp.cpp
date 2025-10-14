#include "block_gemm_omp.h"

#if defined(__GNUC__) || defined(__clang__)
  #define RESTRICT __restrict__
#elif defined(_MSC_VER)
  #define RESTRICT __restrict
#else
  #define RESTRICT
#endif

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {

    const size_t N = static_cast<size_t>(n);
    std::vector<float> C(N * N, 0.0f);

    const size_t bs = 16;

    const float* RESTRICT Adata = a.data();
    const float* RESTRICT Bdata = b.data();
    float* RESTRICT Cdata = C.data();

    #pragma omp parallel for collapse(2) schedule(static) \
        default(none) shared(Adata,Bdata,Cdata,N,bs)
    for (size_t ii = 0; ii < N; ii += bs) {
        for (size_t jj = 0; jj < N; jj += bs) {
            for (size_t kk = 0; kk < N; kk += bs) {
                size_t i_end = std::min(ii + bs, N);
                size_t k_end = std::min(kk + bs, N);
                size_t j_end = std::min(jj + bs, N);

                for (size_t i = ii; i < i_end; ++i) {
                    size_t rowA = i * N;
                    size_t rowC = i * N;
                    for (size_t k = kk; k < k_end; ++k) {
                        float aik = Adata[rowA + k];
                        const float* RESTRICT bptr = Bdata + k * N + jj;
                        float* RESTRICT cptr = Cdata + rowC + jj;

                        for (size_t j = 0; j < j_end - jj; ++j) {
                            cptr[j] += aik * bptr[j];
                        }
                    }
                }
            }
        }
    }

    return C;
}
