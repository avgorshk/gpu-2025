#include "gelu_cuda.h"

#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <cstring> 
#include <cmath>

#define CUDA_CHECK(call)                                                       
    do {                                                                       
        cudaError_t err = (call);                                              
        if (err != cudaSuccess) {                                              
            throw std::runtime_error(std::string("CUDA error: ") +            
                                     cudaGetErrorString(err));                
        }                                                                      
    } while (0)

constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
constexpr float GELU_CONST = 0.044715f;

__global__ void gelu_kernel(const float* __restrict__ input,
                            float* __restrict__ output,
                            size_t n) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x = input[idx];
    float x3 = x * x * x;
    float t = SQRT_2_OVER_PI * (x + GELU_CONST * x3);

    float tanh_t;
    if (t >= 0.0f) {
        // tanh(t) = (1 - exp(-2t)) / (1 + exp(-2t))
        float z = __expf(-2.0f * t);
        tanh_t = (1.0f - z) / (1.0f + z);
    } else {
        // tanh(t) = (exp(2t) - 1) / (exp(2t) + 1)
        float z = __expf(2.0f * t);
        tanh_t = (z - 1.0f) / (z + 1.0f);
    }

    output[idx] = 0.5f * x * (1.0f + tanh_t);
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const size_t n = input.size();
    if (n == 0) return {};

    const size_t bytes = n * sizeof(float);

    float* d_input = nullptr;
    float* d_output = nullptr;

    float* pinned_in = nullptr;
    float* pinned_out = nullptr;

    cudaStream_t stream = nullptr;

    try {

        CUDA_CHECK(cudaMalloc(&d_input, bytes));
        CUDA_CHECK(cudaMalloc(&d_output, bytes));

        CUDA_CHECK(cudaMallocHost(&pinned_in, bytes)); 
        CUDA_CHECK(cudaMallocHost(&pinned_out, bytes)); 

        CUDA_CHECK(cudaStreamCreate(&stream));

        std::memcpy(pinned_in, input.data(), bytes);

        CUDA_CHECK(cudaMemcpyAsync(d_input, pinned_in, bytes,
                                   cudaMemcpyHostToDevice, stream));

        int blockSize = 256;
        int minGridSize = 0;
        CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                                      (void*)gelu_kernel, 0, 0));
        if (blockSize <= 0) blockSize = 256;
        if (blockSize > 1024) blockSize = 1024;

        const int grid = static_cast<int>((n + blockSize - 1) / blockSize);

        gelu_kernel<<<grid, blockSize, 0, stream>>>(d_input, d_output, n);

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyAsync(pinned_out, d_output, bytes,
                                   cudaMemcpyDeviceToHost, stream));

        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::vector<float> result(n);
        std::memcpy(result.data(), pinned_out, bytes);

        CUDA_CHECK(cudaStreamDestroy(stream));
        CUDA_CHECK(cudaFreeHost(pinned_in));
        CUDA_CHECK(cudaFreeHost(pinned_out));
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));

        return result;
    } catch (...) {

        if (stream) cudaStreamDestroy(stream);
        if (pinned_in) cudaFreeHost(pinned_in);
        if (pinned_out) cudaFreeHost(pinned_out);
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
        throw; 
    }
}
