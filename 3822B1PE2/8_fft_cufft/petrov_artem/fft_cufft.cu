#include "fft_cufft.h"
#include <cufft.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
        } \
    } while(0)

#define CUFFT_CHECK(call) \
    do { \
        cufftResult stat = call; \
        if (stat != CUFFT_SUCCESS) { \
            throw std::runtime_error("cuFFT error"); \
        } \
    } while(0)

__global__ void normalize_kernel(cufftComplex* data, int n, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        float scale = 1.0f / static_cast<float>(n);
        data[idx].x *= scale;
        data[idx].y *= scale;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    if (input.empty() || batch <= 0) {
        return std::vector<float>();
    }

    int total_floats = static_cast<int>(input.size());
    if (total_floats % (2 * batch) != 0) {
        throw std::invalid_argument("Input size must be divisible by 2*batch");
    }
    
    int n = total_floats / (2 * batch);
    
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, n, CUFFT_C2C, batch));
    
    CUFFT_CHECK(cufftSetAutoAllocation(plan, 1));
    
    size_t bytes = total_floats * sizeof(float);
    
    float* pinned_output = nullptr;
    CUDA_CHECK(cudaMallocHost(&pinned_output, bytes));
    
    cudaStream_t compute_stream, copy_stream;
    CUDA_CHECK(cudaStreamCreate(&compute_stream));
    CUDA_CHECK(cudaStreamCreate(&copy_stream));
    
    CUFFT_CHECK(cufftSetStream(plan, compute_stream));
    
    cufftComplex* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    
    CUDA_CHECK(cudaMemcpyAsync(d_data, input.data(), bytes, 
                               cudaMemcpyHostToDevice, copy_stream));
    
    CUDA_CHECK(cudaStreamSynchronize(copy_stream));
    
    CUFFT_CHECK(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
    
    CUFFT_CHECK(cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE));
    
    int total_complex = n * batch;
    const int block_size = 256;
    int grid_size = (total_complex + block_size - 1) / block_size;
    
    normalize_kernel<<<grid_size, block_size, 0, compute_stream>>>(
        d_data, n, total_complex);
    
    CUDA_CHECK(cudaMemcpyAsync(pinned_output, d_data, bytes, 
                               cudaMemcpyDeviceToHost, compute_stream));
    
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));
    
    std::vector<float> output(total_floats);
    std::copy(pinned_output, pinned_output + total_floats, output.begin());
    
    CUDA_CHECK(cudaFreeHost(pinned_output));
    CUDA_CHECK(cudaFree(d_data));
    
    CUDA_CHECK(cudaStreamDestroy(compute_stream));
    CUDA_CHECK(cudaStreamDestroy(copy_stream));
    
    CUFFT_CHECK(cufftDestroy(plan));
    
    return output;
}