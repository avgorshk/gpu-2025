#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>

__global__ void gemm_kernel(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n || col >= n) return;

    float sum = 0.0f;
    int k = 0;

    for (; k + 3 < n; k += 4) {
        sum += A[row * n + k + 0] * B[(k + 0) * n + col];
        sum += A[row * n + k + 1] * B[(k + 1) * n + col];
        sum += A[row * n + k + 2] * B[(k + 2) * n + col];
        sum += A[row * n + k + 3] * B[(k + 3) * n + col];
    }

    for (; k < n; ++k)
        sum += A[row * n + k] * B[k * n + col];

    C[row * n + col] = sum;
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& A,
                                 const std::vector<float>& B,
                                 int n) {
    size_t bytes = static_cast<size_t>(n) * n * sizeof(float);
    float *dA, *dB, *dC;

    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);

    cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(32, 32);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    gemm_kernel<<<grid, block>>>(dA, dB, dC, n);
    cudaDeviceSynchronize();

    std::vector<float> C(n * n);
    cudaMemcpy(C.data(), dC, bytes, cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return C;
}
