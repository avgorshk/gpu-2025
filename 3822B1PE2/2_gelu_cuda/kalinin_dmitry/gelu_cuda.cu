#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>

#define SQRT_2_PI 0.7978845608f
#define COEFF 0.044715f
#define HALF 0.5f
#define TWO 2.0f

__global__ void gelu_kernel_optimized(const float* __restrict__ input, 
                                     float* __restrict__ output, 
                                     size_t size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int idx = tid; idx < size; idx += stride) {
        float x = input[idx];
        float x2 = x * x;
        float x3 = x2 * x;
        float inner = SQRT_2_PI * (x + COEFF * x3);
        
        float exp_2z = __expf(TWO * inner);
        float inv_exp_2z_plus_one = __fdividef(1.0f, exp_2z + 1.0f);
        float tanh_val = 1.0f - 2.0f * inv_exp_2z_plus_one;
        
        output[idx] = HALF * x * (1.0f + tanh_val);
    }
}

static float* d_input = nullptr;
static float* d_output = nullptr;
static cudaStream_t stream = nullptr;
static size_t allocated_size = 0;
static bool initialized = false;

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    size_t size = input.size();
    if (size == 0) return std::vector<float>();
    
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
    
    cudaMemcpyAsync(d_input, input.data(), bytes, cudaMemcpyHostToDevice, stream);
    
    const int blockSize = 512;
    const int numBlocks = (size + blockSize - 1) / blockSize;
    gelu_kernel_optimized<<<numBlocks, blockSize, 0, stream>>>(d_input, d_output, size);
    
    std::vector<float> result(size);
    cudaMemcpyAsync(result.data(), d_output, bytes, cudaMemcpyDeviceToHost, stream);
    
    cudaStreamSynchronize(stream);
    
    return result;
}

