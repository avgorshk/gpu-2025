#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>

__global__ void KernelMatrixMul(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if ((row >= N) || (col >= N)) return;

    float acc = 0.0f;
    for (int k = 0; k < N; ++k) {
        acc += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    int totalElems = n * n;
    int bytesSize = totalElems * sizeof(float);
    float *devA, *devB, *devC;
    std::vector<float> result(totalElems);

    cudaMalloc(&devA, bytesSize);
    cudaMalloc(&devB, bytesSize);
    cudaMalloc(&devC, bytesSize);
    cudaMemcpy(devA, a.data(), bytesSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devB, b.data(), bytesSize, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x,
                 (n + blockDim.y - 1) / blockDim.y);
    KernelMatrixMul<<<gridDim, blockDim>>>(devA, devB, devC, n);

    cudaMemcpy(result.data(), devC, bytesSize, cudaMemcpyDeviceToHost);

    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);

    return result;
}
