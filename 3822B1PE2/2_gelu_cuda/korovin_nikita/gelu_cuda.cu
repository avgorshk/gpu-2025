#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>

__constant__ float kInvSqrtPi2_c = 0.7978845608028654f;
__constant__ float kCubicCoeff_c = 0.044715f;


__global__ void gelu_kernel_exp_identity(const float* __restrict__ input, float* __restrict__ output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float x3 = x * x * x;
        float t = kInvSqrtPi2_c * fmaf(kCubicCoeff_c, x3, x);
        float neg_2t = -2.0f * t;
        float exp_neg_2t = __expf(neg_2t); 
        float tanh_val = (1.0f - exp_neg_2t) / (1.0f + exp_neg_2t);
        output[idx] = fmaf(0.5f * x, tanh_val, 0.5f * x);
    }
}


std::vector<float> GeluCUDA(const std::vector<float>& input) {
    size_t n = input.size();
    if (n == 0) return {};
    
    std::vector<float> output(n);
    size_t bytes = n * sizeof(float);
    float *d_input = nullptr, *d_output = nullptr;
    
    cudaMalloc((void**)&d_input, bytes);
    cudaMalloc((void**)&d_output, bytes);

    cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice);
    int minGridSize, blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, gelu_kernel_exp_identity, 0, 0);
    if (blockSize > 1024) blockSize = 1024; 
    int numBlocks = (n + blockSize - 1) / blockSize;
    gelu_kernel_exp_identity<<<numBlocks, blockSize>>>(d_input, d_output, n);

    cudaDeviceSynchronize(); 
    cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}