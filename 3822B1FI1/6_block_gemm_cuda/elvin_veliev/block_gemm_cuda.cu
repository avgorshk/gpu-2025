#include "block_gemm_cuda.h"

#include <cuda_runtime.h>

#define SIZE 16

__global__ void kernel(const float* A, const float* B, float* C, int n) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= n) return;
    
    __shared__ float s_A[SIZE*SIZE];
    __shared__ float s_B[SIZE*SIZE];

    float res = 0.0f;

    for (int k = 0; k < (n + SIZE - 1) / SIZE; ++k) {

        s_A[threadIdx.y*SIZE+threadIdx.x] = A[row * n + k * SIZE + threadIdx.x];
        s_B[threadIdx.y*SIZE+threadIdx.x] = B[(k * SIZE + threadIdx.y) * n + col];
        
        __syncthreads();
        for (int t = 0; t < SIZE; ++t) {
            res += s_A[threadIdx.y*SIZE+t] * s_B[t*SIZE+threadIdx.x];
        }
        __syncthreads();  
    }
    C[row * n + col] = res;
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {
    int memory = n * n * sizeof(float);
    float* A, *B, *C;
	std::vector<float> result(n * n);
    
    cudaMalloc(&A, memory);
    cudaMalloc(&B, memory);
    cudaMalloc(&C, memory);

    cudaMemcpy(A, a.data(), memory, cudaMemcpyHostToDevice);
    cudaMemcpy(B, b.data(), memory, cudaMemcpyHostToDevice);
    
    int grid = (n + SIZE - 1) / SIZE;
    dim3 gridSize(grid, grid);
    dim3 blockSize(SIZE, SIZE);
    
    kernel<<<gridSize, blockSize>>> (A, B, C, n);
    cudaMemcpy(result.data(), C, memory, cudaMemcpyDeviceToHost);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return result;
}