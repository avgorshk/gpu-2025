#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>

#define TILE_SIZE 32
#define SUB_TILE 4

__global__ void matrixMultiplyKernel(const float* A, const float* B, float* C, int N) {
    int blockX = blockIdx.x, blockY = blockIdx.y;
    int threadX = threadIdx.x, threadY = threadIdx.y;

    float partialSum[SUB_TILE][SUB_TILE] = {0};
    int startRow = blockY * TILE_SIZE;
    int startCol = blockX * TILE_SIZE;
    int innerRow = threadY * SUB_TILE;
    int innerCol = threadX * SUB_TILE;
    
    __shared__ float sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedB[TILE_SIZE][TILE_SIZE];
    
    int totalBlocks = (N + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int tileK = 0; tileK < totalBlocks; tileK++) {
        for (int i = 0; i < SUB_TILE; i++) {
            for (int j = 0; j < SUB_TILE; j++) {
                int localRow = innerRow + i;
                int localCol = innerCol + j;
                
                int globalRowA = startRow + localRow;
                int globalColA = tileK * TILE_SIZE + localCol;
                if ((globalRowA < N) && (globalColA < N))
                    sharedA[localRow][localCol] = A[globalRowA * N + globalColA];
                else
                    sharedA[localRow][localCol] = 0.0f;
                    
                int globalRowB = tileK * TILE_SIZE + localRow;
                int globalColB = startCol + localCol;
                if ((globalRowB < N) && (globalColB < N))
                    sharedB[localRow][localCol] = B[globalRowB * N + globalColB];
                else
                    sharedB[localRow][localCol] = 0.0f;
            }
        }
        
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; k++) {
            for (int i = 0; i < SUB_TILE; i++) {
                float aVal = sharedA[innerRow + i][k];
                for (int j = 0; j < SUB_TILE; j++) {
                    partialSum[i][j] += aVal * sharedB[k][innerCol + j];
                }
            }
        }
        
        __syncthreads();
    }

    for (int i = 0; i < SUB_TILE; i++) {
        for (int j = 0; j < SUB_TILE; j++) {
            int globalRow = startRow + innerRow + i;
            int globalCol = startCol + innerCol + j;
            if ((globalRow < N) && (globalCol < N)) {
                C[globalRow * N + globalCol] = partialSum[i][j];
            }
        }
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& matA,
                                 const std::vector<float>& matB,
                                 int N) {
    float *devA, *devB, *devC;
    int totalElems = N * N;
    std::vector<float> result(totalElems, 0.0f);
    size_t bytes = totalElems * sizeof(float);

    cudaMalloc(&devA, bytes);
    cudaMalloc(&devB, bytes);
    cudaMalloc(&devC, bytes);
    
    cudaMemcpy(devA, matA.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(devB, matB.data(), bytes, cudaMemcpyHostToDevice);
    
    int threadsPerSide = TILE_SIZE / SUB_TILE;
    dim3 blockConfig(threadsPerSide, threadsPerSide);
    dim3 gridConfig((N + TILE_SIZE - 1) / TILE_SIZE,
                    (N + TILE_SIZE - 1) / TILE_SIZE);

    matrixMultiplyKernel<<<gridConfig, blockConfig>>>(devA, devB, devC, N);
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(result.data(), devC, bytes, cudaMemcpyDeviceToHost);
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);

    return result;
}
