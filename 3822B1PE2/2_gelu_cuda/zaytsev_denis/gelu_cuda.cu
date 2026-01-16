#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>

__constant__ float c_sqrt_2_over_pi = 0.7978845608028654f;
__constant__ float c_gelu_alpha = 0.044715f;

__device__ inline float fast_exp_neg_2x(float x) {
    return __expf(-2.0f * x);
}

__global__ void gelu_kernel(const float* __restrict__ input,
                           float* __restrict__ output,
                           int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float x = input[idx];
        float x3 = x * x * x;
        
        float inner = c_sqrt_2_over_pi * (x + c_gelu_alpha * x3);
        float exp_term = fast_exp_neg_2x(inner);
        float tanh_approx = 2.0f / (1.0f + exp_term) - 1.0f;
        
        output[idx] = 0.5f * x * (1.0f + tanh_approx);
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int n = static_cast<int>(input.size());
    std::vector<float> output(n);
    
    if (n == 0) return output;
    
    float *d_input = nullptr, *d_output = nullptr;
    size_t bytes = n * sizeof(float);
    
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    
    cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice);
    
    const int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    gelu_kernel<<<numBlocks, blockSize>>>(d_input, d_output, n);
    
    cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    
    return output;
}