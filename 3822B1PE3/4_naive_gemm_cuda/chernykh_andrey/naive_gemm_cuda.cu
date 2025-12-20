#include "naive_gemm_cuda.h"
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


__global__ void NaiveGemmKernel(
    const float *a,
    const float *b,
    float *c,
    int n
) {
    int row_index = blockIdx.y * blockDim.y + threadIdx.y;
    int col_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_index >= n || col_index >= n) {
        return;
    }

    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < n; k++) {
        sum += a[row_index * n + k] * b[k * n + col_index];
    }

    c[row_index * n + col_index] = sum;
}

std::vector<float> NaiveGemmCUDA(
    const std::vector<float> &a,
    const std::vector<float> &b,
    int n
) {
    if (n <= 0) {
        return {};
    }

    size_t size = n * n;
    size_t data_size = size * sizeof(float);
    std::vector<float> c(size);

    float *d_a = nullptr;
    cudaMalloc(&d_a, data_size);
    float *d_b = nullptr;
    cudaMalloc(&d_b, data_size);
    float *d_c = nullptr;
    cudaMalloc(&d_c, data_size);

    cudaMemcpy(d_a, a.data(), data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), data_size, cudaMemcpyHostToDevice);
    dim3 block(32, 32);
    dim3 grid(
        (n + block.x - 1) / block.x,
        (n + block.y - 1) / block.y
    );
    NaiveGemmKernel<<<grid, block>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    cudaMemcpy(c.data(), d_c, data_size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return c;
}
