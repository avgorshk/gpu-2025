#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>

__global__ void gemm_kernel(const float* A, const float* B_transposed, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        int base_a = row * n;
        int base_b = col * n;

        int k = 0;
        for (; k <= n - 4; k += 4) {
            sum += A[base_a + k] * B_transposed[base_b + k];
            sum += A[base_a + k + 1] * B_transposed[base_b + k + 1];
            sum += A[base_a + k + 2] * B_transposed[base_b + k + 2];
            sum += A[base_a + k + 3] * B_transposed[base_b + k + 3];
        }
        for (; k < n; ++k) {
            sum += A[base_a + k] * B_transposed[base_b + k];
        }

        C[base_a + col] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    size_t bytes = n * n * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, a.data(), bytes, cudaMemcpyHostToDevice);

    std::vector<float> b_transposed(n * n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            b_transposed[j * n + i] = b[i * n + j];

    cudaMemcpy(d_B, b_transposed.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, n);

    std::vector<float> C(n * n);
    cudaMemcpy(C.data(), d_C, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return C;
}
