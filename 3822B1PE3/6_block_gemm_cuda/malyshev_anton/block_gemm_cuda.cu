#include "block_gemm_cuda.h"
#include <vector>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>

__device__ __forceinline__ void computeBlockProduct(float* sA, float* sB, 
                                                      int tx, int ty, int bs, float& acc) {
    #pragma unroll 4
    for (int k = 0; k < bs; k++) {
        acc += sA[tx * bs + k] * sB[k * bs + ty];
    }
}

__global__ void blockGemmKernel(const float* A, const float* B, float* C, int n, int bs) {
    extern __shared__ float smem[];
    float* sA = smem;
    float* sB = &smem[bs * bs];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float acc = 0.0f;

    int gridSize = n / bs;
    for (int m = 0; m < gridSize; m++) {
        sA[tx * bs + ty] = A[(by * bs + tx) * n + (m * bs + ty)];
        sB[tx * bs + ty] = B[(m * bs + tx) * n + (bx * bs + ty)];

        __syncthreads();
        computeBlockProduct(sA, sB, tx, ty, bs, acc);
        __syncthreads();
    }

    C[(by * bs + tx) * n + (bx * bs + ty)] = acc;
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

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int bs_mem = sqrtf(prop.sharedMemPerBlock / (2.0f * sizeof(float)));
    int bs_th= sqrtf(prop.maxThreadsPerBlock);
    
    int max_bs = std::min(bs_mem, bs_th);
    int bs = 1;
    while (bs * 2 <= max_bs && n % (bs * 2) == 0) {
        bs <<= 1;
    }

    dim3 threads(bs, bs);
    int grid = (n + bs - 1) / bs;
    dim3 blocks(grid, grid);
    size_t smemSize = 2 * bs * bs * sizeof(float);
    
    blockGemmKernel<<<blocks, threads, smemSize>>>(d_A, d_B, d_C, n, bs);

    cudaDeviceSynchronize();
    cudaMemcpy(c.data(), d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return c;
}