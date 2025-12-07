#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>

inline void ensureCudaSuccess(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA error (" << file << ":" << line << "): "
                  << cudaGetErrorString(code) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
#define CHECK_CUDA(call) ensureCudaSuccess((call), __FILE__, __LINE__)
static constexpr int TILE_DIM = 16;
__global__ void tiledMatmulKernel(const float* __restrict__ matrixA,
                                  const float* __restrict__ matrixB,
                                  float* __restrict__ matrixC,
                                  int N) {
    unsigned int globalRow = blockIdx.y * TILE_DIM + threadIdx.y;
    unsigned int globalCol = blockIdx.x * TILE_DIM + threadIdx.x;
    unsigned int localRow = threadIdx.y;
    unsigned int localCol = threadIdx.x;
    float partialSum = 0.0f;
    __shared__ float tileA[TILE_DIM][TILE_DIM];
    __shared__ float tileB[TILE_DIM][TILE_DIM];
    int numTiles = (N + TILE_DIM - 1) / TILE_DIM;
    for (int tile = 0; tile < numTiles; ++tile) {
        unsigned int aRow = globalRow;
        unsigned int aCol = tile * TILE_DIM + localCol;
        if (aRow < N && aCol < N) {
            tileA[localRow][localCol] = matrixA[aRow * N + aCol];
        } else {
            tileA[localRow][localCol] = 0.0f;
        }
        unsigned int bRow = tile * TILE_DIM + localRow;
        unsigned int bCol = globalCol;
        if (bRow < N && bCol < N) {
            tileB[localRow][localCol] = matrixB[bRow * N + bCol];
        } else {
            tileB[localRow][localCol] = 0.0f;
        }
        __syncthreads();
        for (int k = 0; k < TILE_DIM; ++k) {
            partialSum += tileA[localRow][k] * tileB[k][localCol];
        }
        __syncthreads();
    }
    if (globalRow < N && globalCol < N) {
        matrixC[globalRow * N + globalCol] = partialSum;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& hostA,
                                 const std::vector<float>& hostB,
                                 int N) {
    size_t totalBytes = static_cast<size_t>(N) * N * sizeof(float);
    float *devA = nullptr, *devB = nullptr, *devC = nullptr;
    CHECK_CUDA(cudaMalloc(&devA, totalBytes));
    CHECK_CUDA(cudaMalloc(&devB, totalBytes));
    CHECK_CUDA(cudaMalloc(&devC, totalBytes));
    CHECK_CUDA(cudaMemcpy(devA, hostA.data(), totalBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(devB, hostB.data(), totalBytes, cudaMemcpyHostToDevice));
    dim3 blockSize(TILE_DIM, TILE_DIM);
    dim3 gridSize((N + TILE_DIM - 1) / TILE_DIM,
                  (N + TILE_DIM - 1) / TILE_DIM);
    tiledMatmulKernel<<<gridSize, blockSize>>>(devA, devB, devC, N);
    CHECK_CUDA(cudaGetLastError());
    std::vector<float> hostResult(N * N);
    CHECK_CUDA(cudaMemcpy(hostResult.data(), devC, totalBytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(devA));
    CHECK_CUDA(cudaFree(devB));
    CHECK_CUDA(cudaFree(devC));
    return hostResult;
}