#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <cstring>

#define TILE_SIZE 16

__global__ void blockGemmKernel(const float* a, const float* b, float* c, int n) {
    __shared__ float sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedB[TILE_SIZE][TILE_SIZE];
    
    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;
    const int threadRow = threadIdx.y;
    const int threadCol = threadIdx.x;
    
    const int row = blockRow * TILE_SIZE + threadRow;
    const int col = blockCol * TILE_SIZE + threadCol;
    
    float sum = 0.0f;
    
    const int numTiles = n / TILE_SIZE;
    
    for (int tile = 0; tile < numTiles; ++tile) {
        const int aRow = row;
        const int aCol = tile * TILE_SIZE + threadCol;
        const int bRow = tile * TILE_SIZE + threadRow;
        const int bCol = col;
        
        sharedA[threadRow][threadCol] = a[aRow * n + aCol];
        sharedB[threadRow][threadCol] = b[bRow * n + bCol];
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += sharedA[threadRow][k] * sharedB[k][threadCol];
        }
        
        __syncthreads();
    }
    
    c[row * n + col] = sum;
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
 const std::vector<float>& b,
 int n) {
    const size_t size = n * n * sizeof(float);
    
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice);
    
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim(n / TILE_SIZE, n / TILE_SIZE);
    
    blockGemmKernel<<<gridDim, blockDim>>>(d_a, d_b, d_c, n);
    
    cudaDeviceSynchronize();
    
    std::vector<float> c(n * n);
    cudaMemcpy(c.data(), d_c, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return c;
}

