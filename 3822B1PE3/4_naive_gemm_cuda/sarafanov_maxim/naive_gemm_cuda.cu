#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>

__global__
void naive_gemm_kernel(const float* __restrict__ A,
                       const float* __restrict__ B_T,
                       float* __restrict__ C,
                       int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n || col >= n) return;

    const float* arow = &A[row * n];
    const float* brow = &B_T[col * n];

    float sum = 0.0f;

    int k = 0;

    for (; k <= n - 4; k += 4) {
        float4 va = *(reinterpret_cast<const float4*>(arow + k));
        float4 vb = *(reinterpret_cast<const float4*>(brow + k));

        sum += va.x * vb.x;
        sum += va.y * vb.y;
        sum += va.z * vb.z;
        sum += va.w * vb.w;
    }

    for (; k < n; k++) {
        sum += arow[k] * brow[k];
    }

    C[row * n + col] = sum;
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n)
{
    if (n == 0) return {};

    const size_t bytes = sizeof(float) * n * n;

    float* dA = nullptr;
    float* dB_T = nullptr;
    float* dC = nullptr;

    cudaMallocAsync(&dA,   bytes, 0);
    cudaMallocAsync(&dB_T, bytes, 0);
    cudaMallocAsync(&dC,   bytes, 0);

    std::vector<float> bT(n * n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            bT[j*n + i] = b[i*n + j];

    cudaMemcpyAsync(dA,   a.data(),  bytes, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(dB_T, bT.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x,
              (n + block.y - 1) / block.y);

    naive_gemm_kernel<<<grid, block>>>(dA, dB_T, dC, n);

    std::vector<float> c(n * n);
    cudaMemcpyAsync(c.data(), dC, bytes, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cudaFree(dA);
    cudaFree(dB_T);
    cudaFree(dC);

    return c;
}
