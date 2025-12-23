#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <stdexcept>

__global__ void gemm_naive_kernel(const float *__restrict__ a,
                                  const float *__restrict__ b,
                                  float *__restrict__ c,
                                  int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n || col >= n)
        return;

    float sum = 0.0f;
    for (int k = 0; k < n; ++k)
    {
        sum += a[row * n + k] * b[k * n + col];
    }
    c[row * n + col] = sum;
}

std::vector<float> NaiveGemmCUDA(const std::vector<float> &a,
                                 const std::vector<float> &b,
                                 int n)
{
    if (n == 0)
        return std::vector<float>();

    size_t bytes = n * n * sizeof(float);
    std::vector<float> c(n * n);

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    const dim3 block_size(16, 16);
    const dim3 grid_size((n + block_size.x - 1) / block_size.x,
                         (n + block_size.y - 1) / block_size.y);

    gemm_naive_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);

    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        throw std::runtime_error("CUDA kernel failed: " + std::string(cudaGetErrorString(err)));
    }

    return c;
}