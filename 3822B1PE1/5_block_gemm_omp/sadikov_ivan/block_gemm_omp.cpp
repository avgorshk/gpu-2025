#include "block_gemm_omp.h"

#include <cmath>
#include <omp.h>

void matrixMultiplication(const std::vector<float>& a,
						  const std::vector<float>& b,
						  std::vector<float>& result,
					      int iIndex,
						  int jIndex,
						  int blockSize,
						  int size)
{
	const int iBarier = std::min(iIndex + blockSize, size);
	const int jBarier = std::min(jIndex + blockSize, size);

	for (int i = iIndex; i < iBarier; ++i)
	{
		auto* currentRow = &result[i * size];
		for (int j = jIndex; j < jBarier; ++j)
		{
			auto* currentColumn = &b[j * size];
			float element = a[i * size + j];
			for (int k = 0; k < size; ++k)
			{
				currentRow[k] += element * currentColumn[k];
			}
		}
	}
}


std::vector<float> BlockGemmOMP(const std::vector<float>& a, const std::vector<float>& b, int n)
{
	int blockSize = std::sqrt(n);
	std::vector<float> result(n * n);
#pragma omp parallel
	{
#pragma omp for
		for (int i = 0; i < n; i += blockSize)
		{
			for (int j = 0; j < n; j += blockSize)
			{
				matrixMultiplication(a, b, result, i, j, blockSize, n);
			}
		}
	}
	return result;
}