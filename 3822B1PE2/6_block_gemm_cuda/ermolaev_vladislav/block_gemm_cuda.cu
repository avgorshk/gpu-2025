#include "block_gemm_cuda.h"
#include <vector>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>

int getBlockSize() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int maxThreadsPerBlock = prop.maxThreadsPerBlock;
    int sharedMemPerBlock = prop.sharedMemPerBlock;
    int maxBlockSizeFromSharedMem = sqrtf(sharedMemPerBlock / (2.0f * sizeof(float)));
    int maxBlockSizeFromThreads = sqrtf(maxThreadsPerBlock);
    
    int blockSize = 1;
    while (blockSize * 2 <= std::min(maxBlockSizeFromSharedMem, maxBlockSizeFromThreads)) {
        blockSize <<= 1;
    }
    return blockSize;
}

__device__ __forceinline__ void computeBlockProduct(float* sharedA, float* sharedB, 
                                                      int row, int col, int blockSize, float& sum) {
    #pragma unroll 4
    for (int i = 0; i < blockSize; i++) {
        sum += sharedA[row * blockSize + i] * sharedB[i * blockSize + col];
    }
}

__global__ void blockGemmKernel(const float* A, const float* B, float* C, int n, int blockSize) {
    extern __shared__ float sharedMem[];
    float* sharedA = sharedMem;
    float* sharedB = &sharedMem[blockSize * blockSize];

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int row = threadIdx.y;
    int col = threadIdx.x;

    float sum = 0.0f;

    int numBlocks = n / blockSize;
    for (int bk = 0; bk < numBlocks; bk++) {
        sharedA[row * blockSize + col] = A[(blockRow * blockSize + row) * n + (bk * blockSize + col)];
        sharedB[row * blockSize + col] = B[(bk * blockSize + row) * n + (blockCol * blockSize + col)];

        __syncthreads();
        computeBlockProduct(sharedA, sharedB, row, col, blockSize, sum);
        __syncthreads();
    }

    C[(blockRow * blockSize + row) * n + (blockCol * blockSize + col)] = sum;
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    std::vector<float> c(n * n, 0.0f);
    
    float *d_A, *d_B, *d_C;
    size_t size = n * n * sizeof(float);
    
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    cudaMemcpy(d_A, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b.data(), size, cudaMemcpyHostToDevice);
    
    int blockSize = getBlockSize();
    while (n % blockSize != 0 && blockSize > 1) {
        blockSize /= 2;
    }

    dim3 threadsPerBlock(blockSize, blockSize);
    int num_blocks = (n + blockSize - 1) / blockSize;
    dim3 numBlocks(num_blocks, num_blocks);
    size_t sharedMemSize = 2 * blockSize * blockSize * sizeof(float);
    
    blockGemmKernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(d_A, d_B, d_C, n, blockSize);

    cudaDeviceSynchronize();
    cudaMemcpy(c.data(), d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return c;
}