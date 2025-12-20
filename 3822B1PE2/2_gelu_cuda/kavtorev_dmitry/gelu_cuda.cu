#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>

constexpr float SQRT_2_PI = 0.7978845608f;
constexpr float COEFF = 0.044715f;
constexpr float HALF = 0.5f;
constexpr float TWO = 2.0f;

__global__ void gelu_kernel(const float* input, float* output, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float x = input[idx];
        float x3 = x * x * x;
        float inner = SQRT_2_PI * (x + COEFF * x3);
        
        float exp_2z = __expf(TWO * inner);
        float tanh_val = (exp_2z - 1.0f) / (exp_2z + 1.0f);
        
        output[idx] = HALF * x * (1.0f + tanh_val);
    }
}

static float* d_input = nullptr;
static float* d_output = nullptr;
static size_t allocated_size = 0;

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    size_t size = input.size();
    if (size == 0) return std::vector<float>();
    
    size_t bytes = size * sizeof(float);
    
    if (d_input == nullptr || allocated_size < size) {
        if (d_input != nullptr) {
            cudaFree(d_input);
            cudaFree(d_output);
        }
        cudaMalloc(&d_input, bytes);
        cudaMalloc(&d_output, bytes);
        allocated_size = size;
    }
    
    cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice);
    
    const int blockSize = 256;
    const int numBlocks = (size + blockSize - 1) / blockSize;
    gelu_kernel<<<numBlocks, blockSize>>>(d_input, d_output, size);
    
    std::vector<float> result(size);
    cudaMemcpy(result.data(), d_output, bytes, cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();
    
    return result;
}

