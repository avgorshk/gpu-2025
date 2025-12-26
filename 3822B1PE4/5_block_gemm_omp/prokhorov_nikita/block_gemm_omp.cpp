#include "block_gemm_omp.h"
#include <omp.h>
#include <vector>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
	const std::vector<float>& b,
	int n) {
	std::vector<float> c(n * n, 0.0f);

	const int BLOCK_SIZE = 64;

#pragma omp parallel for collapse(2)
	for (int i_block = 0; i_block < n; i_block += BLOCK_SIZE) {
		for (int j_block = 0; j_block < n; j_block += BLOCK_SIZE) {

			for (int k_block = 0; k_block < n; k_block += BLOCK_SIZE) {
				int i_end = std::min(i_block + BLOCK_SIZE, n);
				int j_end = std::min(j_block + BLOCK_SIZE, n);
				int k_end = std::min(k_block + BLOCK_SIZE, n);

				for (int i = i_block; i < i_end; ++i) {
					for (int k = k_block; k < k_end; ++k) {
						float a_ik = a[i * n + k];
#pragma omp simd
						for (int j = j_block; j < j_end; ++j) {
							c[i * n + j] += a_ik * b[k * n + j];
						}
					}
				}
			}
		}
	}

	return c;
}