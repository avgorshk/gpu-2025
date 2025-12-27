#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>

__global__ void BlockMulKernel(const float *A, const float *B, float *C,
                                     int n)
{
    const int tile_width = 16;

    __shared__ float shared_A[tile_width][tile_width];
    __shared__ float shared_B[tile_width][tile_width];

    int thr_X = threadIdx.x;
    int thr_Y = threadIdx.y;

    int row = blockIdx.y * tile_width + thr_Y;
    int col = blockIdx.x * tile_width + thr_X;

    float collected_sum = 0.0f;

    for (int tile = 0; tile < (n + tile_width - 1) / tile_width; ++tile)
    {
        if (row < n && (tile * tile_width + thr_X) < n)
        {
            shared_A[thr_Y][thr_X] = A[row * n + (tile * tile_width + thr_X)];
        }
        else
        {
            shared_A[thr_Y][thr_X] = 0.0f;
        }

        if ((tile * tile_width + thr_Y) < n && col < n)
        {
            shared_B[thr_Y][thr_X] = B[(tile * tile_width + thr_Y) * n + col];
        }
        else
        {
            shared_B[thr_Y][thr_X] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < tile_width; ++k)
        {
            collected_sum += shared_A[thr_Y][k] * shared_B[k][thr_X];
        }

        __syncthreads();
    }

    if (row < n && col < n)
    {
        C[row * n + col] = collected_sum;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& A,
                                 const std::vector<float>& B,
                                 int n) {
    

    float *d_A, *d_B, *d_C;
    size_t matrixSize = n * n * sizeof(float);
    
    cudaMalloc(&d_A, matrixSize);
    cudaMalloc(&d_B, matrixSize);
    cudaMalloc(&d_C, matrixSize);
    
    cudaMemcpy(d_A, A.data(), matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), matrixSize, cudaMemcpyHostToDevice);
    
    dim3 blockSize(16, 16);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x,
                  (n + blockSize.y - 1) / blockSize.y);
    
    BlockMulKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);

    std::vector<float> C(n * n);
    cudaMemcpy(C.data(), d_C, matrixSize, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return C;
}