#include "gelu_cuda.h"
#include <cuda_runtime.h>

__constant__ float kSqrt2OverPi = 0.7978845608028654f;
__constant__ float kCoeff = 0.044715f;

__global__ void GeluKernel(const float* __restrict__ input, 
                           float* __restrict__ output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float x3 = x * x * x;
        float t = kSqrt2OverPi * fmaf(kCoeff, x3, x);
        float exp_neg2t = __expf(-2.0f * t);
        float tanh_t = (1.0f - exp_neg2t) / (1.0f + exp_neg2t);
        output[idx] = 0.5f * x * (1.0f + tanh_t);
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    size_t n = input.size();
    if (n == 0) return {};
    
    std::vector<float> output(n);
    size_t bytes = n * sizeof(float);
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    
    cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    GeluKernel<<<numBlocks, blockSize>>>(d_input, d_output, n);
    
    cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    return output;
}

