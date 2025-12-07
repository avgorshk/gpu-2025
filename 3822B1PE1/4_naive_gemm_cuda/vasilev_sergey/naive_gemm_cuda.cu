#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>
#include <cstddef>
#include <vector>

namespace
{

    __global__ void naive_gemm_kernel(const float *a,
                                      const float *b,
                                      float *c,
                                      int n)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= n || col >= n)
        {
            return;
        }

        float sum = 0.0f;
        int row_offset = row * n;
        for (int k = 0; k < n; ++k)
        {
            sum += a[row_offset + k] * b[k * n + col];
        }
        c[row_offset + col] = sum;
    }

} // namespace

std::vector<float> NaiveGemmCUDA(const std::vector<float> &a,
                                 const std::vector<float> &b,
                                 int n)
{
    std::size_t nn = static_cast<std::size_t>(n);
    std::vector<float> c(nn * nn);
    if (n <= 0)
    {
        return c;
    }

    std::size_t bytes = nn * nn * sizeof(float);

    float *d_a;
    float *d_b;
    float *d_c;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x,
              (n + block.y - 1) / block.y);

    naive_gemm_kernel<<<grid, block>>>(d_a, d_b, d_c, n);

    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}
