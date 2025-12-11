#include "gelu_cuda.h"
#include <cuda_runtime.h>

__global__ void gelu_vectorized_kernel(const float* input, float* output, int n) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    int stride = gridDim.x * blockDim.x * 2;
    
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    const float half = 0.5f;
    
    for (int i = idx; i < n; i += stride) {
        if (i + 1 < n) {
            float x1 = input[i];
            float x2 = input[i + 1];
            
            float x1_2 = x1 * x1;
            float x2_2 = x2 * x2;
            
            float inner1 = sqrt_2_over_pi * (x1 + coeff * x1_2 * x1);
            float inner2 = sqrt_2_over_pi * (x2 + coeff * x2_2 * x2);
            
            output[i] = half * x1 * (2.0f / (1.0f + __expf(-2.0f * inner1)));
            output[i + 1] = half * x2 * (2.0f / (1.0f + __expf(-2.0f * inner2)));
        } 
        else if (i < n) {
            float x = input[i];
            float x2 = x * x;
            float inner = sqrt_2_over_pi * (x + coeff * x2 * x);
            output[i] = half * x * (2.0f / (1.0f + __expf(-2.0f * inner)));
        }
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int n = input.size();
    std::vector<float> output(n);
    
    if (n == 0) return output;
    
    float *d_input, *d_output;
    size_t size = n * sizeof(float);
    
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    
    cudaMemcpy(d_input, input.data(), size, cudaMemcpyHostToDevice);
    
    const int blockSize = 256;
    
    int numBlocks = (n / 2 + blockSize - 1) / blockSize;
    
    if (numBlocks < 64) numBlocks = 64;
    
    int maxBlocks = 65535;
    if (numBlocks > maxBlocks) numBlocks = maxBlocks;
    
    gelu_vectorized_kernel<<<numBlocks, blockSize>>>(d_input, d_output, n);
    cudaDeviceSynchronize();
    
    cudaMemcpy(output.data(), d_output, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    return output;
}
