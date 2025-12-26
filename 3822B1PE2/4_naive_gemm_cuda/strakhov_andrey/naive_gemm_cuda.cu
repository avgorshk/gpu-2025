#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>

__global__ void kernel(const float *a, const float *b, float *res, int n)
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
    res[i * n + j] = sum;
}


std::vector<float> NaiveGemmCUDA(const std::vector<float> &a,
                                 const std::vector<float> &b,
                                 int n)
{
    int size = n * n * sizeof(float);
    float *local_a, *local_b, *local_res;
    std::vector<float> res(n * n);

    cudaMalloc(&local_a, size);
    cudaMalloc(&local_b, size);
    cudaMalloc(&local_res, size);

    cudaMemcpy(local_a, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(local_b, b.data(), size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    kernel<<<grid, block>>>(local_a, local_b, local_res, n);
    cudaMemcpy(res.data(), local_res, size, cudaMemcpyDeviceToHost);

    cudaFree(local_a);
    cudaFree(local_b);
    cudaFree(local_res);

    return res;
}