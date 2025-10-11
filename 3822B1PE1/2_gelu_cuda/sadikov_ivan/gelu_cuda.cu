#include "gelu_cuda.h"

#include <cuda_runtime.h>
#include <iostream>

__global__ void Calculate(const float* inputData, float* result, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        result[index] = 0.5f * inputData[index] * (1.0f  + 
            tanhf(0.7978f * (inputData[index] + 0.044715f * inputData[index] * inputData[index] * inputData[index])));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) 
{
    auto size {input.size() * sizeof(float)};
    constexpr int threadsCount {256};
    int blocksCount {(static_cast<int>(input.size()) + threadsCount - 1) / threadsCount};

    std::vector<float> result(input.size());

    float * inputData;
    float * resultFromCUDA;

    cudaMalloc(&inputData, size);
    cudaMalloc(&resultFromCUDA, size);

    cudaMemcpy(inputData, input.data(), size, cudaMemcpyHostToDevice);

    Calculate<<<blocksCount, threadsCount>>>(inputData, resultFromCUDA, static_cast<int>(input.size()));

    cudaMemcpy(resultFromCUDA, result.data(), size, cudaMemcpyDeviceToHost);
    
    cudaFree(inputData);
    cudaFree(resultFromCUDA);
    
    return result;
}