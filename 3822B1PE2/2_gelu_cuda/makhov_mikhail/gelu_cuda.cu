#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

constexpr float sqrt_2_over_pi = 0.7978845608028654f;
constexpr float coeff = 0.044715f;
constexpr float half = 0.5f;
constexpr float two = 2.0f;

__global__ void gelu_kernel(const float* __restrict__ input,
                            float* __restrict__ output,
                            int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float x = input[idx];
        float x2 = x * x;
        float x3 = x2 * x;
        float inner = sqrt_2_over_pi * (x + coeff * x3);
        
        float exp_2z = __expf(two * inner);
        float tanh_val = (exp_2z - 1.0f) / (exp_2z + 1.0f);
        
        output[idx] = half * x * (1.0f + tanh_val);
    }
}

static float* d_input = nullptr;
static float* d_output = nullptr;
static cudaStream_t stream = nullptr;
static size_t allocated_size = 0;
static bool initialized = false;

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    size_t size = input.size();
    if (size == 0) {
        return std::vector<float>();
    }
    
    if (!initialized) {
        cudaStreamCreate(&stream);
        initialized = true;
    }
    
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
    
    cudaMemcpyAsync(d_input, input.data(), bytes, 
                    cudaMemcpyHostToDevice, stream);
    
    const int blockSize = 256;
    const int numBlocks = (size + blockSize - 1) / blockSize;
    gelu_kernel<<<numBlocks, blockSize, 0, stream>>>(d_input, d_output, static_cast<int>(size));
    
    cudaGetLastError();
    
    std::vector<float> result(size);
    
    cudaMemcpyAsync(result.data(), d_output, bytes,
                    cudaMemcpyDeviceToHost, stream);
    
    cudaStreamSynchronize(stream);
    
    return result;
}

