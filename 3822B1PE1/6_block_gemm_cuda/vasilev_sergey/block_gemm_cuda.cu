#include "block_gemm_cuda.h"

#include <cuda_runtime.h>
#include <cstddef>
#include <vector>

namespace
{

    constexpr int kBlockSize = 16;

    __global__ void block_gemm_kernel(const float *a,
                                      const float *b,
                                      float *c,
                                      int n)
    {
        __shared__ float as[kBlockSize][kBlockSize];
        __shared__ float bs[kBlockSize][kBlockSize];

        int row = blockIdx.y * kBlockSize + threadIdx.y;
        int col = blockIdx.x * kBlockSize + threadIdx.x;

        float sum = 0.0f;

        for (int t = 0; t < n; t += kBlockSize)
        {
            int a_col = t + threadIdx.x;
            int b_row = t + threadIdx.y;

            if (row < n && a_col < n)
            {
                as[threadIdx.y][threadIdx.x] = a[row * n + a_col];
            }
            else
            {
                as[threadIdx.y][threadIdx.x] = 0.0f;
            }

            if (b_row < n && col < n)
            {
                bs[threadIdx.y][threadIdx.x] = b[b_row * n + col];
            }
            else
            {
                bs[threadIdx.y][threadIdx.x] = 0.0f;
            }

            __syncthreads();

#pragma unroll
            for (int k = 0; k < kBlockSize; ++k)
            {
                sum += as[threadIdx.y][k] * bs[k][threadIdx.x];
            }

            __syncthreads();
        }

        if (row < n && col < n)
        {
            c[row * n + col] = sum;
        }
    }

} // namespace

std::vector<float> BlockGemmCUDA(const std::vector<float> &a,
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

    dim3 block(kBlockSize, kBlockSize);
    dim3 grid((n + block.x - 1) / block.x,
              (n + block.y - 1) / block.y);

    block_gemm_kernel<<<grid, block>>>(d_a, d_b, d_c, n);

    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}
