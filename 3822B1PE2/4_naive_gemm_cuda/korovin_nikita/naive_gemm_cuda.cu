#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath> 
#include <cstdio>
#include <cstddef>

#define BLOCK_SIZE 16 


__global__ void MatrixMulKernel(const float* A, const float* B, float* C, int N) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        const float4* A_vec = reinterpret_cast<const float4*>(A);
        for (int k_vec = 0; k_vec < N / 4; ++k_vec) {
            float4 a_vals = A_vec[row * (N / 4) + k_vec];
            int k_base = k_vec * 4;
            sum += a_vals.x * B[(k_base + 0) * N + col];
            sum += a_vals.y * B[(k_base + 1) * N + col];
            sum += a_vals.z * B[(k_base + 2) * N + col];
            sum += a_vals.w * B[(k_base + 3) * N + col];
        }
        C[row * N + col] = sum;
    }
}


std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n <= 0) {
        return std::vector<float>();
    }

    const size_t matrix_size = static_cast<size_t>(n) * n;
    const size_t size = matrix_size * sizeof(float);
    
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
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); 

    const int KernelTileSide = BLOCK_SIZE;
    dim3 KernelBlock(KernelTileSide, KernelTileSide);
    int GridW = (n + KernelBlock.x - 1) / KernelBlock.x;
    dim3 LaunchGrid(GridW, (n + KernelBlock.y - 1) / KernelBlock.y);
    
    MatrixMulKernel<<<LaunchGrid, KernelBlock>>>(d_A, d_B, d_C, n);

    cudaDeviceSynchronize();

    cudaMemcpy(h_C_result.data(), d_C, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return h_C_result;
}
