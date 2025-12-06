#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

__global__ void gemm_kernel(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= n || col >= n) return;
    
    float sum = 0.0f;

    int k = 0;
    int limit = n - 3;

    for (; k < limit; k += 4) {
        float a0 = A[row * n + k];
        float a1 = A[row * n + k + 1];
        float a2 = A[row * n + k + 2];
        float a3 = A[row * n + k + 3];

        float b0 = B[k * n + col];
        float b1 = B[(k + 1) * n + col];
        float b2 = B[(k + 2) * n + col];
        float b3 = B[(k + 3) * n + col];

        sum += a0 * b0;
        sum += a1 * b1;
        sum += a2 * b2;
        sum += a3 * b3;
    }
    for (; k < n; ++k) {
        sum += A[row * n + k] * B[k * n + col];
    }
    
    C[row * n + col] = sum;
}

static float* d_A = nullptr;
static float* d_B = nullptr;
static float* d_C = nullptr;
static size_t allocated_size = 0;

void free_gpu_memory() {
    if (d_A) cudaFree(d_A);
    if (d_B) cudaFree(d_B);
    if (d_C) cudaFree(d_C);
    
    d_A = d_B = d_C = nullptr;
    allocated_size = 0;
}

bool allocate_gpu_memory(size_t bytes) {
    cudaError_t err;
    
    err = cudaMalloc(&d_A, bytes);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed for A: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    err = cudaMalloc(&d_B, bytes);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed for B: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        d_A = nullptr;
        return false;
    }
    
    err = cudaMalloc(&d_C, bytes);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed for C: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        d_A = d_B = nullptr;
        return false;
    }
    
    allocated_size = bytes;
    return true;
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n <= 0) {
        return std::vector<float>();
    }
    
    const size_t matrix_size = n * n;
    const size_t bytes_needed = matrix_size * sizeof(float);
    
    if (a.size() != matrix_size || b.size() != matrix_size) {
        throw std::runtime_error("Input matrix size mismatch");
    }

    std::vector<float> c(matrix_size, 0.0f);

    if (allocated_size < bytes_needed) {
        free_gpu_memory();
        if (!allocate_gpu_memory(bytes_needed)) {
            throw std::runtime_error("Failed to allocate GPU memory");
        }
    }
    
    cudaError_t err;

    err = cudaMemcpy(d_A, a.data(), bytes_needed, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpy A failed: ") + cudaGetErrorString(err));
    }
    
    err = cudaMemcpy(d_B, b.data(), bytes_needed, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpy B failed: ") + cudaGetErrorString(err));
    }

    const int BLOCK_SIZE = 16;
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    gemm_kernel<<<grid_dim, block_dim>>>(d_A, d_B, d_C, n);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Kernel launch failed: ") + cudaGetErrorString(err));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaDeviceSynchronize failed: ") + cudaGetErrorString(err));
    }

    err = cudaMemcpy(c.data(), d_C, bytes_needed, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpy C failed: ") + cudaGetErrorString(err));
    }
    
    return c;
}

struct CUDACleanup {
    ~CUDACleanup() {
        free_gpu_memory();
    }
};

static CUDACleanup cleanup;