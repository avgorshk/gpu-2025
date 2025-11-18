#include "block_gemm_omp.h"
#include <immintrin.h>
#include <cmath>

using std::vector;

vector<float> BlockGemmOMP(const vector<float>& a,
                           const vector<float>& b,
                           int n) {
    int size = n * n;
    vector<float> ans(size, 0.f);

    int block_n = 32;
    if(n < 64) block_n = n;

    #pragma omp parallel for
    for (int i = 0; i < n; i += block_n) {
        for (int j = 0; j < n; j += block_n) {
            for (int k = 0; k < n; k += block_n) {
                for (int w = k; w < k + block_n; w++) {
                     for (int u = i; u < i + block_n; u++){
                        for (int v = j; v < j + block_n; v++)  {
                            ans[u * n + v] += a[u * n + w] * b[w * n + v];
                        }
                    }
                }
            }
        }
    }
    return ans;
}
