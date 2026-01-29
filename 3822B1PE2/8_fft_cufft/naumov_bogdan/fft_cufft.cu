#include "fft_cufft.h"
#include <cufft.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>

__global__ void normalize_kernel(cufftComplex* data, int n, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        data[idx].x /= n;
        data[idx].y /= n;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    assert(input.size() % (2 * batch) == 0 && "Input size must be divisible by 2*batch");
    int n = input.size() / (2 * batch);
    
    std::vector<float> output(input.size());

    cufftComplex* host_input = nullptr;
    cufftComplex* host_output = nullptr;
    cudaHostAlloc((void**)&host_input, input.size() * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc((void**)&host_output, input.size() * sizeof(float), cudaHostAllocDefault);

    for (size_t i = 0; i < input.size() / 2; ++i) {
        host_input[i].x = input[2 * i];
        host_input[i].y = input[2 * i + 1];
    }

    cufftComplex* device_data = nullptr;
    cudaMalloc((void**)&device_data, input.size() * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(device_data, host_input, 
                   input.size() * sizeof(float), 
                   cudaMemcpyHostToDevice, stream);

    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_C2C, batch);
    cufftSetStream(plan, stream);

    cufftExecC2C(plan, device_data, device_data, CUFFT_FORWARD);

    cufftExecC2C(plan, device_data, device_data, CUFFT_INVERSE);

    int total_elements = n * batch;
    int blockSize = 256;
    int numBlocks = (total_elements + blockSize - 1) / blockSize;
    normalize_kernel<<<numBlocks, blockSize, 0, stream>>>(device_data, n, total_elements);

    cudaMemcpyAsync(host_output, device_data, 
                   input.size() * sizeof(float), 
                   cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    for (size_t i = 0; i < input.size() / 2; ++i) {
        output[2 * i] = host_output[i].x;
        output[2 * i + 1] = host_output[i].y;
    }

    cufftDestroy(plan);
    cudaFree(device_data);
    cudaFreeHost(host_input);
    cudaFreeHost(host_output);
    cudaStreamDestroy(stream);
    
    return output;
}