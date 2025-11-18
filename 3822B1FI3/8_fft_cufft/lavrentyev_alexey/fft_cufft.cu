#include "fft_cufft.h"
#include <cufft.h>

using std::vector;
const int block_size = 32;

__constant__ float normalize;
__constant__ int size;

__global__ void kernel_normilize_data(float *a){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size){
        a[idx] *= normalize;
    }
}

vector<float> FffCUFFT(const vector<float>& input, int batch) {

    int size = (int)(input.size());
	int n = size / (2 * batch);
    float norm = 1.0f / static_cast<float>(n);
    cudaMemcpyToSymbol(normalize, &norm, sizeof(float));
    cudaMemcpyToSymbol(size, &size, sizeof(int));

    cufftComplex* complex;
    cufftHandle handler;
	
    vector<float> ans(size);

	cudaMalloc(&complex,  size * sizeof(float));
	cudaMemcpy(complex, input.data(),  size * sizeof(float), cudaMemcpyHostToDevice);
	cufftPlan1d(&handler, n, CUFFT_C2C, batch);
    cufftExecC2C(handler, complex, complex, CUFFT_FORWARD);
    cufftExecC2C(handler, complex, complex, CUFFT_INVERSE);
	kernel_normilize_data<<<(size + block_size - 1) / block_size, block_size>>> ((float*)(complex));
	cudaMemcpy(ans.data(), complex,  size * sizeof(float), cudaMemcpyDeviceToHost);
	cufftDestroy(handler);
	cudaFree(complex);

	return ans;
}
