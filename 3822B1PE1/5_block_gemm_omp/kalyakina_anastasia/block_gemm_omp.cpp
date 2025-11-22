#include "block_gemm_omp.h"

#include <omp.h>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {

	std::vector<float> c(n * n, 0.0f);

	const int block_size = (omp_get_max_threads() >= 16 && n < 64) ? 8 : 16;
	const int num_blocks = n / block_size;

#pragma omp parallel for collapse(2)
	for (int block_i = 0; block_i < num_blocks; ++block_i) {
		for (int block_j = 0; block_j < num_blocks; ++block_j) {

			std::vector<float> temp_block(block_size * block_size, 0.0f);

			for (int block_k = 0; block_k < num_blocks; ++block_k) {
				const int a_start = block_i * block_size * n + block_k * block_size;
				const int b_start = block_k * block_size * n + block_j * block_size;

				for (int i = 0; i < block_size; ++i) {
					const float* a_ptr = &a[a_start + i * n];
					for (int k = 0; k < block_size; ++k) {
						float a_val = a_ptr[k];
						const float* b_ptr = &b[b_start + k * n];
						float* temp_ptr = &temp_block[i * block_size];

						for (int j = 0; j < block_size; ++j) {
							temp_ptr[j] += a_val * b_ptr[j];
						}
					}
				}
			}

			const int c_start = block_i * block_size * n + block_j * block_size;
			for (int i = 0; i < block_size; ++i) {
				float* c_ptr = &c[c_start + i * n];
				float* temp_ptr = &temp_block[i * block_size];
				for (int j = 0; j < block_size; ++j) {
					c_ptr[j] = temp_ptr[j];
				}
			}
		}
	}

	return c;
}