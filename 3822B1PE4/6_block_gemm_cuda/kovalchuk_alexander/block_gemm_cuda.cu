#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <stdexcept>

constexpr int TILE = 16;

__global__ void block_gemm_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum0 = 0.0f;
    float sum1 = 0.0f;

    int col1 = col + TILE / 2;
    bool has_second = (col1 < n);

    int numTiles = n / TILE;

    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;

        if (row < n && aCol < n) {
            As[threadIdx.y][threadIdx.x] = A[row * n + aCol];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int bCol0 = col;
        int bCol1 = has_second ? col1 : col;

        if (bRow < n && bCol0 < n) {
            Bs[threadIdx.y][threadIdx.x] = B[bRow * n + bCol0];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            float a_val = As[threadIdx.y][k];
            float b0    = Bs[k][threadIdx.x];
            sum0 += a_val * b0;

            if (has_second) {
                int b1_col_local = threadIdx.x + TILE / 2;
                float b1 = Bs[k][b1_col_local];
                sum1 += a_val * b1;
            }
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = sum0;
    }
    if (has_second && row < n) {
        C[row * n + col1] = sum1;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    const size_t nn = static_cast<size_t>(n) * static_cast<size_t>(n);
    if (a.size() != nn || b.size() != nn) {
        throw std::runtime_error("BlockGemmCUDA: wrong input sizes");
    }

    std::vector<float> c(nn);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    cudaError_t err;

    err = cudaMalloc(&d_a, nn * sizeof(float));
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc d_a failed");

    err = cudaMalloc(&d_b, nn * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_a);
        throw std::runtime_error("cudaMalloc d_b failed");
    }

    err = cudaMalloc(&d_c, nn * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        throw std::runtime_error("cudaMalloc d_c failed");
    }

    err = cudaMemcpy(d_a, a.data(), nn * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        throw std::runtime_error("cudaMemcpy H2D a failed");
    }

    err = cudaMemcpy(d_b, b.data(), nn * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        throw std::runtime_error("cudaMemcpy H2D b failed");
    }

    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);

    block_gemm_kernel<<<grid, block>>>(d_a, d_b, d_c, n);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        throw std::runtime_error("cudaDeviceSynchronize failed");
    }

    err = cudaMemcpy(c.data(), d_c, nn * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        throw std::runtime_error("cudaMemcpy D2H c failed");
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}
