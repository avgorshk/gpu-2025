#include "gelu_cuda.h"
#include <cmath>
#include <cuda_runtime.h>

using std::vector;

constexpr int block_size = 256;


__global__ void GeluKernel(const float* in, float* ans, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float var = in[idx];
        float cube = var * var * var;
        float tanh_arg = 0.79788f * (var + 0.044715f * cube);
        ans[idx] = 0.5f * var * (1.f + tanh(tanh_arg));
    }
}

__host__ vector<float> GeluCUDA(const vector<float>& input) {
    int size = input.size(), req_mem = size * sizeof(float);
    float* in, *ans;
	vector<float> result(size);

    cudaMalloc(&in, req_mem);
    cudaMalloc(&ans, req_mem);

    cudaMemcpy(in, input.data(), req_mem, cudaMemcpyKind::cudaMemcpyHostToDevice);
    GeluKernel<<<(size + block_size - 1) / block_size, block_size>>> (in, ans, size);
    
    cudaMemcpy(result.data(), ans, req_mem, cudaMemcpyKind::cudaMemcpyDeviceToHost);

    cudaFree(in);
    cudaFree(ans);

    return result;
}
