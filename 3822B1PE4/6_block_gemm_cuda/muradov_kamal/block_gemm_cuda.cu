#include "block_gemm_cuda.h"

#include <cuda_runtime.h>
#include <vector>

constexpr int kTile = 16;

__global__ void BlockGemmKernel(const float* a, const float* b, float* c, int n) {
    __shared__ float tile_a[kTile][kTile];
    __shared__ float tile_b[kTile][kTile];

    int row = blockIdx.y * kTile + threadIdx.y;
    int col = blockIdx.x * kTile + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < n; t += kTile) {
        int a_col = t + threadIdx.x;
        int b_row = t + threadIdx.y;
        if (row < n && a_col < n) {
            tile_a[threadIdx.y][threadIdx.x] = a[row * n + a_col];
        } else {
            tile_a[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if (col < n && b_row < n) {
            tile_b[threadIdx.y][threadIdx.x] = b[b_row * n + col];
        } else {
            tile_b[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < kTile; ++k) {
            sum += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    std::vector<float> c(static_cast<size_t>(n) * n);
    if (n == 0) {
        return c;
    }

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    size_t bytes = static_cast<size_t>(n) * n * sizeof(float);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(kTile, kTile);
    dim3 grid((n + kTile - 1) / kTile, (n + kTile - 1) / kTile);
    BlockGemmKernel<<<grid, block>>>(d_a, d_b, d_c, n);

    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}
