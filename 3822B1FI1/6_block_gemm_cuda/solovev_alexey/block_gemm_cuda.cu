#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

constexpr int TILE = 16;

__global__ void block_gemm_kernel(const float* A, const float* B, float* C, int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;

    for (int t = 0; t < n; t += TILE) {
        int a_col = t + tx;
        if (row < n && a_col < n)
            As[ty][tx] = A[row * n + a_col];
        else
            As[ty][tx] = 0.0f;

        int b_row = t + ty;
        if (b_row < n && col < n)
            Bs[ty][tx] = B[b_row * n + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {
    if (n <= 0) return {};
    if (a.size() != static_cast<size_t>(n) * static_cast<size_t>(n) ||
        b.size() != static_cast<size_t>(n) * static_cast<size_t>(n)) {
        throw std::invalid_argument("Input sizes do not match n*n");
    }

    size_t bytes = static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(float);
    std::vector<float> c(static_cast<size_t>(n) * static_cast<size_t>(n), 0.0f);

    float* d_A = nullptr, * d_B = nullptr, * d_C = nullptr;

    if (cudaMalloc(&d_A, bytes) != cudaSuccess) throw std::runtime_error("cudaMalloc A failed");
    if (cudaMalloc(&d_B, bytes) != cudaSuccess) { cudaFree(d_A); throw std::runtime_error("cudaMalloc B failed"); }
    if (cudaMalloc(&d_C, bytes) != cudaSuccess) { cudaFree(d_A); cudaFree(d_B); throw std::runtime_error("cudaMalloc C failed"); }

    if (cudaMemcpy(d_A, a.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        throw std::runtime_error("cudaMemcpy H2D A failed");
    }
    if (cudaMemcpy(d_B, b.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        throw std::runtime_error("cudaMemcpy H2D B failed");
    }

    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);

    block_gemm_kernel << <grid, block >> > (d_A, d_B, d_C, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        throw std::runtime_error(std::string("Kernel launch failed: ") + cudaGetErrorString(err));
    }

    if (cudaDeviceSynchronize() != cudaSuccess) {
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        throw std::runtime_error("Kernel execution failed (synchronize)");
    }

    if (cudaMemcpy(c.data(), d_C, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        throw std::runtime_error("cudaMemcpy D2H C failed");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return c;
}