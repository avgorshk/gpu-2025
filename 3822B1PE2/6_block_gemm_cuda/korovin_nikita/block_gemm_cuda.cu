#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstddef>  
#include <algorithm> 
#include <cmath> 

#define BLOCK_SIZE 32 


__global__ void blockGemmKernel(const float* A, const float* B, float* C, int N) {
    const int TILE_SIZE = BLOCK_SIZE;

    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;
    int global_row = blockIdx.y * TILE_SIZE + thread_row;
    int global_col = blockIdx.x * TILE_SIZE + thread_col;

    float Csub = 0.0f;

    int num_blocks = (N + TILE_SIZE - 1) / TILE_SIZE; 
    for (int block_k = 0; block_k < num_blocks; ++block_k) {
        
        int k_coord_start = block_k * TILE_SIZE;
        int global_A_idx = global_row * N + (k_coord_start + thread_col);
        if (global_row < N && (k_coord_start + thread_col) < N) {
            sA[thread_row][thread_col] = A[global_A_idx];
        } 
        else {
            sA[thread_row][thread_col] = 0.0f;
        }

        int global_B_row = k_coord_start + thread_row;
        int global_B_col = global_col;
        int global_B_idx = global_B_row * N + global_B_col;
        
        if (global_B_row < N && global_B_col < N) {
            sB[thread_col][thread_row] = B[global_B_idx]; 
        } 
        else {
            sB[thread_col][thread_row] = 0.0f;
        }
        __syncthreads();
        for (int k_idx = 0; k_idx < TILE_SIZE; ++k_idx) {
            Csub += sA[thread_row][k_idx] * sB[thread_col][k_idx];
        }
        __syncthreads();
    }
    if (global_row < N && global_col < N) {
        C[global_row * N + global_col] = Csub;
    }
}


std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n <= 0) {
        return std::vector<float>();
    }

    size_t matrix_size = static_cast<size_t>(n) * n;
    size_t size = matrix_size * sizeof(float);
    
    if (a.size() != matrix_size || b.size() != matrix_size) {
        return std::vector<float>();
    }

    std::vector<float> h_C_result(matrix_size);
    
    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b.data(), size, cudaMemcpyHostToDevice);
    
    const int KernelTileSide = BLOCK_SIZE; 
    dim3 KernelBlock(KernelTileSide, KernelTileSide); 

    int GridW = (n + KernelBlock.x - 1) / KernelBlock.x;
    dim3 LaunchGrid(GridW, (n + KernelBlock.y - 1) / KernelBlock.y); 
    
    blockGemmKernel<<<LaunchGrid, KernelBlock>>>(d_A, d_B, d_C, n);

    cudaDeviceSynchronize();

    cudaMemcpy(h_C_result.data(), d_C, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return h_C_result;
}
