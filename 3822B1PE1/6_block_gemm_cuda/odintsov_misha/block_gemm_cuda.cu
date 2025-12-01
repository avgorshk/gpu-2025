#include "block_gemm_cuda.h"
#include <vector>
#include <cuda_runtime.h>

#define TILE_DIM 16 

__global__ void blockMatrixMultiplyKernel(float* C, const float* A, const float* B, int n) {
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    __shared__ float sharedA[TILE_DIM][TILE_DIM];
    __shared__ float sharedB[TILE_DIM][TILE_DIM];

    float sum = 0.0f;

    for (int k = 0; k < n / TILE_DIM; ++k) {
        sharedA[threadIdx.y][threadIdx.x] = A[row * n + (k * TILE_DIM + threadIdx.x)];
        sharedB[threadIdx.y][threadIdx.x] = B[(k * TILE_DIM + threadIdx.y) * n + col];

        __syncthreads();

        for (int i = 0; i < TILE_DIM; ++i) {
            sum += sharedA[threadIdx.y][i] * sharedB[i][threadIdx.x];
        }

        __syncthreads();
    }

    C[row * n + col] = sum;
   
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) {
    size_t len = n * n;

    std::vector<float> output(len);
    float* matrix_a;
    float* matrix_b;
    float* ans;

    cudaMalloc((void**)&matrix_a, len * sizeof(float));
    cudaMalloc((void**)&matrix_b, len * sizeof(float));
    cudaMalloc((void**)&ans, len * sizeof(float));

    cudaMemcpy(matrix_a, a.data(), len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(matrix_b, b.data(), len * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(TILE_DIM, TILE_DIM); 
    dim3 numBlocks(n / TILE_DIM, n / TILE_DIM); 

    blockMatrixMultiplyKernel<<<numBlocks, blockSize>>>(ans, matrix_a, matrix_b, n);

    cudaDeviceSynchronize();

    cudaMemcpy(output.data(), ans, len * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(matrix_a);
    cudaFree(matrix_b);
    cudaFree(ans);

    return output;
}
