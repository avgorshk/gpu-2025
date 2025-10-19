#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>

#define BLOCK_SIZE 16


__global__ void MatrixMulKernel(const float* A, const float* B_T, float* C, int N) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B_T[col * N + k];
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
    if (a.size() != matrix_size || b.size() != matrix_size) {
        return std::vector<float>();
    }

    std::vector<float> b_transposed(matrix_size);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            b_transposed[i * n + j] = b[j * n + i];
        }
    }

    std::vector<float> h_C(matrix_size);
    float *d_A, *d_B_T, *d_C;
    const size_t bytes = matrix_size * sizeof(float);
    
    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B_T, bytes);
    cudaMalloc((void**)&d_C, bytes);

    cudaMemcpy(d_A, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_T, b_transposed.data(), bytes, cudaMemcpyHostToDevice); 
    
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    const int gridDimSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 gridDim(gridDimSize, gridDimSize);

    MatrixMulKernel<<<gridDim, blockDim>>>(d_A, d_B_T, d_C, n);
    
    cudaGetLastError(); 

    cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();

    cudaFree(d_A);
    cudaFree(d_B_T);
    cudaFree(d_C);

    return h_C;
}
