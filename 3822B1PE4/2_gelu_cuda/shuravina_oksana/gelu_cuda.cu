#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

const float SQRT_2_OVER_PI = 0.7978845608028654f;
const float COEFFICIENT = 0.044715f;

__global__ void gelu_kernel_fast(const float* __restrict__ input, 
                                float* __restrict__ output, 
                                int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float x = input[idx];
        float x_cubed = x * x * x;
        float inner = SQRT_2_OVER_PI * (x + COEFFICIENT * x_cubed);

        float exp_2inner = expf(2.0f * inner);
        float tanh_approx = (exp_2inner - 1.0f) / (exp_2inner + 1.0f);
        
        output[idx] = 0.5f * x * (1.0f + tanh_approx);
    }
}

static float* d_input = nullptr;
static float* d_output = nullptr;
static size_t current_buffer_size = 0;
static cudaStream_t compute_stream;

static bool InitializeCUDAResources(size_t required_size) {
    static bool initialized = false;
    
    if (!initialized) {
        cudaStreamCreate(&compute_stream);
        initialized = true;
    }

    if (d_input == nullptr || current_buffer_size < required_size) {
        if (d_input != nullptr) {
            cudaFree(d_input);
            cudaFree(d_output);
        }
        
        cudaMalloc(&d_input, required_size);
        cudaMalloc(&d_output, required_size);
        current_buffer_size = required_size;
        
        if (d_input == nullptr || d_output == nullptr) {
            std::cerr << "Failed to allocate GPU memory" << std::endl;
            return false;
        }
    }
    
    return true;
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int size = input.size();
    std::vector<float> result(size);
    
    if (size == 0) {
        return result;
    }
    
    size_t required_bytes = size * sizeof(float);

    if (!InitializeCUDAResources(required_bytes)) {
        std::cerr << "GPU initialization failed, using CPU fallback" << std::endl;
        const float sqrt_2_over_pi = 0.7978845608028654f;
        const float coefficient = 0.044715f;
        
        for (int i = 0; i < size; ++i) {
            float x = input[i];
            float x_cubed = x * x * x;
            float inner = sqrt_2_over_pi * (x + coefficient * x_cubed);
            float exp_2inner = expf(2.0f * inner);
            float tanh_approx = (exp_2inner - 1.0f) / (exp_2inner + 1.0f);
            result[i] = 0.5f * x * (1.0f + tanh_approx);
        }
        return result;
    }

    cudaMemcpyAsync(d_input, input.data(), required_bytes, 
                   cudaMemcpyHostToDevice, compute_stream);

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    gelu_kernel_fast<<<numBlocks, blockSize, 0, compute_stream>>>(d_input, d_output, size);

    cudaError_t kernel_error = cudaGetLastError();
    if (kernel_error != cudaSuccess) {
        std::cerr << "CUDA kernel error: " << cudaGetErrorString(kernel_error) << std::endl;
        return result;
    }

    cudaMemcpyAsync(result.data(), d_output, required_bytes, 
                   cudaMemcpyDeviceToHost, compute_stream);

    cudaStreamSynchronize(compute_stream);
    
    return result;
}

void CleanupGeluCUDA() {
    if (d_input != nullptr) {
        cudaFree(d_input);
        cudaFree(d_output);
        d_input = nullptr;
        d_output = nullptr;
        current_buffer_size = 0;
    }
    
    cudaStreamDestroy(compute_stream);
}
