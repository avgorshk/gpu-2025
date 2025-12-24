#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>

__global__ void matrixMultiplication(const float* firstMatrix,
                                     const float* secondMatrix,
                                     float* resultMatrix,
                                     int size)
{
    int rowNum = blockIdx.y * blockDim.y + threadIdx.y;
    int columnNum = blockIdx.x * blockDim.x + threadIdx.x;

    if (rowNum < size && columnNum < size)
    {
        auto* element = &firstMatrix[rowNum * size];
        float result{ 0.0f };

        for (int k = 0; k < size; ++k)
        {
            result += element[k] * secondMatrix[k * size + columnNum];
        }
        resultMatrix[rowNum * size + columnNum] = result;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n)
{
    constexpr int threadInLine{ 16 };
    const int elementsCount{ n * n };
    const int bytesCount{ elementsCount * static_cast<int>(sizeof(float)) };

    std::vector<float> result(elementsCount);

    float* firstMatrix;
    float* secondMatrix;
    float* resultMatrix;

    cudaMalloc(&firstMatrix, bytesCount);
    cudaMalloc(&secondMatrix, bytesCount);
    cudaMalloc(&resultMatrix, bytesCount);

    cudaMemcpy(firstMatrix, a.data(), bytesCount, cudaMemcpyHostToDevice);
    cudaMemcpy(secondMatrix, b.data(), bytesCount, cudaMemcpyHostToDevice);

    dim3 threadsCount(threadInLine, threadInLine);
    dim3 blocksCount((n + threadInLine - 1) / threadInLine,
                     (n + threadInLine - 1) / threadInLine);

    matrixMultiplication <<<blocksCount, threadsCount >>> (firstMatrix, secondMatrix, resultMatrix, n);

    cudaMemcpy(result.data(), resultMatrix, bytesCount, cudaMemcpyDeviceToHost);

    cudaFree(firstMatrix);
    cudaFree(secondMatrix);
    cudaFree(resultMatrix);

    return result;
}