#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <stdexcept>

__global__ void naive_gemm_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col_start = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

    if (row >= n || col_start >= n) return;

    float sum0 = 0.0f;
    float sum1 = 0.0f;

    int col1 = col_start + 1;
    bool has_second = (col1 < n);

    for (int k = 0; k < n; ++k) {
        float a_val = A[row * n + k];
        float b0 = B[k * n + col_start];
        sum0 += a_val * b0;

        if (has_second) {
            float b1 = B[k * n + col1];
            sum1 += a_val * b1;
        }
    }

    C[row * n + col_start] = sum0;
    if (has_second) {
        C[row * n + col1] = sum1;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    const size_t nn = static_cast<size_t>(n) * static_cast<size_t>(n);
    if (a.size() != nn || b.size() != nn) {
        throw std::runtime_error("NaiveGemmCUDA: wrong input sizes");
    }

    std::vector<float> c(nn);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    cudaError_t err;

    err = cudaMalloc(&d_a, nn * sizeof(float));
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMalloc d_a failed");
    }
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
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        throw std::runtime_error("cudaMemcpy H2D a failed");
    }
    err = cudaMemcpy(d_b, b.data(), nn * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        throw std::runtime_error("cudaMemcpy H2D b failed");
    }

    dim3 block(16, 16);
    dim3 grid((n + block.x * 2 - 1) / (block.x * 2),
              (n + block.y - 1) / block.y);

    naive_gemm_kernel<<<grid, block>>>(d_a, d_b, d_c, n);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        throw std::runtime_error("cudaDeviceSynchronize failed");
    }

    err = cudaMemcpy(c.data(), d_c, nn * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        throw std::runtime_error("cudaMemcpy D2H c failed");
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}
