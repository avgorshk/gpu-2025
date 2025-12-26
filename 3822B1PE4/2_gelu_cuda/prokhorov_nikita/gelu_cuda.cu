#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>

__global__ void gelu_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    float x = input[idx];
    float sigmoid = 1.0f / (1.0f + expf(-1.702f * x));
    output[idx] = x * sigmoid;
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int n = input.size();
    if (n == 0) return {};
    
    std::vector<float> result(n);
    size_t size = n * sizeof(float);
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    
    cudaMemcpy(d_input, input.data(), size, cudaMemcpyHostToDevice);
    
    int block = 256;
    int grid = (n + block - 1) / block;
    gelu_kernel<<<grid, block>>>(d_input, d_output, n);
    
    cudaMemcpy(result.data(), d_output, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    return result;
}