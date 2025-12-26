#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>
#include <stdexcept>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        throw std::runtime_error(cudaGetErrorString(err));         \
    }                                                              \
} while(0)

__global__ void naiveGemmKernel(const float* __restrict__ a,
                                 const float* __restrict__ b,
                                 float* __restrict__ c,
                                 int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n <= 0) {
        return {};
    }

    const size_t matrixSize = static_cast<size_t>(n) * n;
    if (a.size() != matrixSize || b.size() != matrixSize) {
        throw std::invalid_argument("Matrix sizes do not match n*n");
    }

    const size_t bytes = matrixSize * sizeof(float);
    std::vector<float> result(matrixSize);

    float* deviceA = nullptr;
    float* deviceB = nullptr;
    float* deviceC = nullptr;

    CUDA_CHECK(cudaMalloc(&deviceA, bytes));
    CUDA_CHECK(cudaMalloc(&deviceB, bytes));
    CUDA_CHECK(cudaMalloc(&deviceC, bytes));

    CUDA_CHECK(cudaMemcpy(deviceA, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceB, b.data(), bytes, cudaMemcpyHostToDevice));

    const int blockDim = 16;
    dim3 threadsPerBlock(blockDim, blockDim);
    dim3 blocksPerGrid((n + blockDim - 1) / blockDim, (n + blockDim - 1) / blockDim);

    naiveGemmKernel<<<blocksPerGrid, threadsPerBlock>>>(deviceA, deviceB, deviceC, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(result.data(), deviceC, bytes, cudaMemcpyDeviceToHost));

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    return result;
}
