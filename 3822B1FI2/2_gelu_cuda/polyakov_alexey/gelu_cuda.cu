#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
constexpr float COEF = 0.044715f;

__global__ void gelu_overlap_kernel(const float* input, float* output, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float x3 = x * x * x;
        float inner = SQRT_2_OVER_PI * (x + COEF * x3);

        float exp_val = __expf(2.0f * inner);
        float tanh_approx = 1.0f - 2.0f / (exp_val + 1.0f);
        
        output[idx] = 0.5f * x * (1.0f + tanh_approx);
    }
}

static float* d_input_overlap = nullptr;
static float* d_output_overlap = nullptr;
static cudaStream_t stream1, stream2;
static bool streams_initialized = false;

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    size_t size = input.size();
    if (size == 0) return std::vector<float>();

    if (!streams_initialized) {
        cudaStreamCreate(&stream1);
        cudaStreamCreate(&stream2);
        streams_initialized = true;
    }
    
    size_t bytes = size * sizeof(float);
    std::vector<float> output(size);

    if (!d_input_overlap) cudaMalloc(&d_input_overlap, bytes);
    if (!d_output_overlap) cudaMalloc(&d_output_overlap, bytes);

    size_t half_size = size / 2;
    size_t second_half_size = size - half_size;

    cudaMemcpyAsync(d_input_overlap, input.data(), half_size * sizeof(float), 
                   cudaMemcpyHostToDevice, stream1);
    
    cudaMemcpyAsync(d_input_overlap + half_size, input.data() + half_size, 
                   second_half_size * sizeof(float), cudaMemcpyHostToDevice, stream2);
    
    int blockSize = 256;
    int numBlocks1 = (half_size + blockSize - 1) / blockSize;
    int numBlocks2 = (second_half_size + blockSize - 1) / blockSize;
    
    gelu_overlap_kernel<<<numBlocks1, blockSize, 0, stream1>>>(d_input_overlap, d_output_overlap, half_size);
    gelu_overlap_kernel<<<numBlocks2, blockSize, 0, stream2>>>(d_input_overlap + half_size, 
                                                              d_output_overlap + half_size, second_half_size);
    
    cudaMemcpyAsync(output.data(), d_output_overlap, half_size * sizeof(float), 
                   cudaMemcpyDeviceToHost, stream1);
    cudaMemcpyAsync(output.data() + half_size, d_output_overlap + half_size, 
                   second_half_size * sizeof(float), cudaMemcpyDeviceToHost, stream2);
    
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    
    return output;
}