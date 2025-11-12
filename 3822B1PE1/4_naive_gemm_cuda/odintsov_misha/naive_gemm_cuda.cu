#include "naive_gemm_cuda.h"
#include <vector>
#include <cuda_runtime.h>


__global__ void matrixMultiplyKernel(float* C, const float* A, const float* B, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;

        for (int k = 0; k < n; k += 4) {
            sum += A[row * n + k] * B[k * n + col] +
                   A[row * n + k + 1] * B[(k + 1) * n + col] +
                   A[row * n + k + 2] * B[(k + 2) * n + col] +
                   A[row * n + k + 3] * B[(k + 3) * n + col];
        }

        C[row * n + col] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                  const std::vector<float>& b,
                                  int n) {
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

    dim3 blockSize(8, 8); 
    dim3 numBlocks(n / blockSize.x, n / blockSize.y);  


    matrixMultiplyKernel<<<numBlocks, blockSize>>>(ans, matrix_a, matrix_b, n);


    cudaDeviceSynchronize();

    cudaMemcpy(output.data(), ans, len * sizeof(float), cudaMemcpyDeviceToHost);


    cudaFree(matrix_a);
    cudaFree(matrix_b);
    cudaFree(ans);

    return output;
}
