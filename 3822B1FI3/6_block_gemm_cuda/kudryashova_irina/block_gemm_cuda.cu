#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_SIZE 32

__global__ void block_gemm_kernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ c,
                                  int n)
{
    __shared__ float a_tile[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float b_tile[BLOCK_SIZE][BLOCK_SIZE + 1];

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int row_idx = blockIdx.y * BLOCK_SIZE + ty;
    const int col_idx = blockIdx.x * BLOCK_SIZE + tx;

    float acc = 0.0f;
    const int num_tiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        const int a_col_base = t * BLOCK_SIZE;
        const int b_row_base = t * BLOCK_SIZE;

        if (row_idx < n) {
            int col4 = a_col_base + (tx << 2); // tx * 4
            if (tx < (BLOCK_SIZE >> 2) && (col4 + 3) < n) {
                const float4* src = reinterpret_cast<const float4*>(
                    &a[row_idx * n + col4]);
                float4 v = *src;
                a_tile[ty][(tx << 2) + 0] = v.x;
                a_tile[ty][(tx << 2) + 1] = v.y;
                a_tile[ty][(tx << 2) + 2] = v.z;
                a_tile[ty][(tx << 2) + 3] = v.w;
            } else if (tx < BLOCK_SIZE) {
                int col = a_col_base + tx;
                a_tile[ty][tx] = (col < n) ? a[row_idx * n + col] : 0.0f;
            }
        } else if (tx < BLOCK_SIZE) {
            a_tile[ty][tx] = 0.0f;
        }

        if (col_idx < n) {
            int row_b = b_row_base + ty;
            if (row_b < n) {
                int col4 = col_idx & ~3;
                if (((col_idx % 4) == 0) && (col4 + 3) < n) {
                    const float4* src = reinterpret_cast<const float4*>(
                        &b[row_b * n + col4]);
                    float4 v = *src;
                    int local_tx = tx;
                    if (local_tx % 4 == 0) {
                        b_tile[ty][local_tx + 0] = v.x;
                        b_tile[ty][local_tx + 1] = v.y;
                        b_tile[ty][local_tx + 2] = v.z;
                        b_tile[ty][local_tx + 3] = v.w;
                    } else {
                        b_tile[ty][tx] = b[row_b * n + col_idx];
                    }
                } else {
                    b_tile[ty][tx] = b[row_b * n + col_idx];
                }
            } else {
                b_tile[ty][tx] = 0.0f;
            }
        } else {
            if (tx < BLOCK_SIZE) b_tile[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            acc += a_tile[ty][k] * b_tile[k][tx];
        }

        __syncthreads();
    }

    if (row_idx < n && col_idx < n) {
        c[row_idx * n + col_idx] = acc;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n)
{
    const size_t bytes = static_cast<size_t>(n) * n * sizeof(float);
    std::vector<float> c(n * n);

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    cudaHostRegister(const_cast<float*>(a.data()), bytes, 0);
    cudaHostRegister(const_cast<float*>(b.data()), bytes, 0);
    cudaHostRegister(c.data(),                     bytes, 0);

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    cudaMallocAsync(&d_a, bytes, stream);
    cudaMallocAsync(&d_b, bytes, stream);
    cudaMallocAsync(&d_c, bytes, stream);

    cudaMemcpyAsync(d_a, a.data(), bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, b.data(), bytes, cudaMemcpyHostToDevice, stream);

    dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks_per_grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
                         (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    block_gemm_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(d_a, d_b, d_c, n);

    cudaMemcpyAsync(c.data(), d_c, bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFreeAsync(d_a, stream);
    cudaFreeAsync(d_b, stream);
    cudaFreeAsync(d_c, stream);

    cudaHostUnregister(const_cast<float*>(a.data()));
    cudaHostUnregister(const_cast<float*>(b.data()));
    cudaHostUnregister(c.data());

    cudaStreamDestroy(stream);
    return c;
}