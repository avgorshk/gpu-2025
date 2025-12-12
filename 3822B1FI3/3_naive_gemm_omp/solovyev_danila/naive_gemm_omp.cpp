
#include "naive_gemm_omp.h"

#include <cmath>
#include <omp.h>
#include <vector>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> output(n*n);
    #pragma omp parallel for
    for (int i=0;i<n;i++){
      for (int j=0;j<n;j++){
        for (int k=0;k<n;k++){
          output[i*n+j]+=a[i*n+k]*b[k*n+j];
        }
      }
    }
    return output;
}
