#include "naive_gemm_omp.h"

#include <omp.h>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(n * n, 0.0f);

#pragma omp parallel for collapse(2)
	for (int i = 0; i < n; i++) {
		for (int k = 0; k < n; k++) {

			float a_current = a[i * n + k];

			float* c_ptr = &c[i * n];
			const float* b_ptr = &b[k * n];

			for (int j = 0; j < n; j++) {
				c_ptr[j] += a_current * b_ptr[j];
			}
		}
	}

	return c;
}