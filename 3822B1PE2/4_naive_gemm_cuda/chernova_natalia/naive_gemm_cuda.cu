#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

__global__ void MultMatrEl(const float *a, const float *b, float *c, int n)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if ((i >= n) || (j >= n))
        return;

    float sum = 0.0;
    for (int k = 0; k < n; k++)
    {
        sum += a[i * n + k] * b[k * n + j];
    }
    c[i * n + j] = sum;
}

std::vector<float> NaiveGemmCUDA(const std::vector<float> &a,
                                 const std::vector<float> &b,
                                 int n)
{
    if (a.size() != static_cast<size_t>(n * n) ||
        b.size() != static_cast<size_t>(n * n))
    {
        std::cerr << "Error: Matrix sizes don't match!" << std::endl;
        return std::vector<float>();
    }

    int num_elements = n * n;
    int byte_size = num_elements * sizeof(float);
    float *d_a, *d_b, *d_c;
    std::vector<float> c(num_elements);

    cudaMalloc(&d_a, byte_size);
    cudaMalloc(&d_b, byte_size);
    cudaMalloc(&d_c, byte_size);
    cudaMemcpy(d_a, a.data(), byte_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), byte_size, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (n + blockSize.y - 1) / blockSize.y);
    MultMatrEl<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

    cudaMemcpy(c.data(), d_c, byte_size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}