#ifndef __GEMM_CUBLAS_H
#define __GEMM_CUBLAS_H

#include <vector>

std::vector<float> GemmCUBLAS(const std::vector<float>& matrix_a,
	const std::vector<float>& matrix_b,
	int matrix_size);

#endif // __GEMM_CUBLAS_H