#include "fft_cufft.h"

#include <cufft.h>
#include <cuda_runtime.h>

__global__ void normalize_kernel(cufftComplex* data, int n, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i].x *= scale;
        data[i].y *= scale;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    std::vector<float> output(input.size());

	int n = input.size() / (2 * batch);
	size_t complex_size = n * batch;

	cufftComplex* d_data;
	size_t size = complex_size * sizeof(cufftComplex);

	cudaMalloc(&d_data, size);
	cudaMemcpy(d_data, input.data(), size, cudaMemcpyHostToDevice);

	cufftHandle plan;
	cufftPlan1d(&plan, n, CUFFT_C2C, batch);
	cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
	cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);

	int blockSize = 256;
	int numBlocks = (complex_size + blockSize - 1) / blockSize;

	normalize_kernel << <numBlocks, blockSize >> > (d_data, complex_size, 1.0f / n);

	cudaMemcpy(output.data(), d_data, size, cudaMemcpyDeviceToHost);

	cufftDestroy(plan);
	cudaFree(d_data);

	return output;
}