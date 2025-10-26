#include <cstdlib>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "block_gemm_cuda.h"

#define BLOCK_SIZE 32

__global__ void BlockGemmKernel(const float *matrixA, const float *matrixB, float *matrixC, const size_t matrixSize)
{
    __shared__ float tileA[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float tileB[BLOCK_SIZE][BLOCK_SIZE + 1];

    const int threadCol = threadIdx.x;
    const int threadRow = threadIdx.y;
    const int globalRow = blockIdx.y * BLOCK_SIZE + threadRow;
    const int globalCol = blockIdx.x * BLOCK_SIZE + threadCol;
    const int rowStride = globalRow * matrixSize;
    const bool isValidRow = globalRow < matrixSize;
    const bool isValidCol = globalCol < matrixSize;

    float accumulator = 0.0f;

    for (int tileIndex = 0; tileIndex < matrixSize / BLOCK_SIZE; ++tileIndex)
    {
        const int tileOffset = tileIndex * BLOCK_SIZE;
        const int elementA = rowStride + tileOffset + threadCol;
        const int elementB = (tileOffset + threadRow) * matrixSize + globalCol;

        tileA[threadRow][threadCol] = (isValidRow && (tileOffset + threadCol) < matrixSize) ? __ldg(&matrixA[elementA]) : 0.0f;
        tileB[threadRow][threadCol] = (isValidCol && (tileOffset + threadRow) < matrixSize) ? __ldg(&matrixB[elementB]) : 0.0f;

        __syncthreads();

#pragma unroll
        for (int innerIndex = 0; innerIndex < BLOCK_SIZE; ++innerIndex)
        {
            accumulator += tileA[threadRow][innerIndex] * tileB[innerIndex][threadCol];
        }

        __syncthreads();
    }

    if (isValidRow && isValidCol)
    {
        matrixC[rowStride + globalCol] = accumulator;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float> &matrixA,
                                 const std::vector<float> &matrixB, int matrixSize)
{
    cudaDeviceProp deviceProperties;
    cudaGetDeviceProperties(&deviceProperties, 0);

    const auto elementCount = matrixSize * matrixSize;
    std::vector<float> resultMatrix(elementCount);
    const auto byteSize = elementCount * sizeof(float);

    dim3 threadBlock(BLOCK_SIZE, BLOCK_SIZE);
    auto blocksPerDimension = (matrixSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 gridDimensions(blocksPerDimension, blocksPerDimension);

    float *deviceA = nullptr;
    cudaMalloc(&deviceA, byteSize);

    float *deviceB = nullptr;
    cudaMalloc(&deviceB, byteSize);

    float *deviceC = nullptr;
    cudaMalloc(&deviceC, byteSize);

    cudaMemcpy(deviceA, matrixA.data(), byteSize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, matrixB.data(), byteSize, cudaMemcpyHostToDevice);

    BlockGemmKernel<<<gridDimensions, threadBlock>>>(deviceA, deviceB, deviceC, matrixSize);

    cudaDeviceSynchronize();
    cudaMemcpy(resultMatrix.data(), deviceC, byteSize, cudaMemcpyDeviceToHost);

    cudaFree(deviceC);
    cudaFree(deviceB);
    cudaFree(deviceA);

    return resultMatrix;
}