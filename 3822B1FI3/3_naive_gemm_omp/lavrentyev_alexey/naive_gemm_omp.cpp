#include "naive_gemm_omp.h"
#include <omp.h>

using std::vector;

vector<float> NaiveGemmOMP(const vector<float>& a,
                           const vector<float>& b,
                           int n) {
    int size = n * n;
    vector<float> ans(size, 0.0f), transpose(size);
    
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            transpose[j * n + i] = b[i * n + j];
        }
    }
    
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                ans[i * n + j] += a[i * n + k] * transpose[j * n + k];
            }
        }
    }

    return ans;
}
