//
// Created by korablev-vm on 27.09.2025.
//

#include "gelu_cuda.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

constexpr float SQRT_2_OVER_PI = 0.7978845608f;
constexpr float GELU_COEFF     = 0.044715f;

__global__ void GeluCUDA_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x  = data[idx];
        float x2 = x * x;
        float inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x * x2);

        data[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    int threadsPerBlock = deviceProp.maxThreadsPerBlock;
    int blocksNum = (input.size() + threadsPerBlock - 1) / threadsPerBlock;

    std::vector<float> output(input);

    float* d_ptr = nullptr;
    cudaMalloc(&d_ptr, sizeof(float) * input.size());
    cudaMemcpy(d_ptr, output.data(), sizeof(float) * input.size(), cudaMemcpyHostToDevice);

    GeluCUDA_kernel<<<blocksNum, threadsPerBlock>>>(d_ptr, static_cast<int>(input.size()));
    cudaDeviceSynchronize();

    cudaMemcpy(output.data(), d_ptr, sizeof(float) * input.size(), cudaMemcpyDeviceToHost);
    cudaFree(d_ptr);

    return output;
}
