#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

__global__ void GeluKernelFast(const float* input, float* output, int n) {
    const float a = 1.702f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        output[idx] = x * (0.5f + 0.5f * tanhf(x * a * 0.5f));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int n = input.size();
    std::vector<float> output(n);
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    cudaMemcpyAsync(d_input, input.data(), n * sizeof(float), cudaMemcpyHostToDevice, stream);
    
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    GeluKernelFast<<<numBlocks, blockSize, 0, stream>>>(d_input, d_output, n);
    cudaMemcpyAsync(output.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost, stream);
    
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    return output;
}