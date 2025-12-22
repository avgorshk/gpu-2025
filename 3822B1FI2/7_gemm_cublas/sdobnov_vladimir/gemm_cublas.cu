#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <stdexcept>
#include <vector>

static float* d_A = nullptr;
static float* d_B = nullptr;
static float* d_C = nullptr;
static size_t allocated_size = 0;
static cublasHandle_t cublas_handle = nullptr;

void init_cublas() {
    if (cublas_handle == nullptr) {
        cublasStatus_t status = cublasCreate(&cublas_handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cuBLAS initialization failed");
        }

        #if CUDA_VERSION >= 11000
        cublasMath_t cublas_math_mode = CUBLAS_TF32_TENSOR_OP_MATH;
        cublasSetMathMode(cublas_handle, cublas_math_mode);
        #endif
    }
}

void cleanup() {
    if (d_A) cudaFree(d_A);
    if (d_B) cudaFree(d_B);
    if (d_C) cudaFree(d_C);
    d_A = d_B = d_C = nullptr;
    allocated_size = 0;
    
    if (cublas_handle) {
        cublasDestroy(cublas_handle);
        cublas_handle = nullptr;
    }
}

void allocate_gpu_memory(size_t bytes) {
    if (allocated_size >= bytes) return;
    
    cleanup();
    
    cudaError_t cuda_err;
    cuda_err = cudaMalloc(&d_A, bytes);
    if (cuda_err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory for A");
    }
    
    cuda_err = cudaMalloc(&d_B, bytes);
    if (cuda_err != cudaSuccess) {
        cudaFree(d_A);
        throw std::runtime_error("Failed to allocate GPU memory for B");
    }
    
    cuda_err = cudaMalloc(&d_C, bytes);
    if (cuda_err != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        throw std::runtime_error("Failed to allocate GPU memory for C");
    }
    
    allocated_size = bytes;
}

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    // Проверка входных данных
    if (n <= 0) return std::vector<float>();
    
    const size_t matrix_size = n * n;
    const size_t bytes_needed = matrix_size * sizeof(float);
    
    if (a.size() != matrix_size || b.size() != matrix_size) {
        throw std::runtime_error("Input matrix size mismatch");
    }

    init_cublas();

    allocate_gpu_memory(bytes_needed);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cublasSetStream(cublas_handle, stream);
    
    std::vector<float> a_transposed(matrix_size);
    std::vector<float> b_transposed(matrix_size);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            a_transposed[j * n + i] = a[i * n + j];  // A^T
            b_transposed[j * n + i] = b[i * n + j];  // B^T
        }
    }
    
    cudaMemcpyAsync(d_A, a_transposed.data(), bytes_needed, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, b_transposed.data(), bytes_needed, cudaMemcpyHostToDevice, stream);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    cublasStatus_t status = cublasSgemm(cublas_handle,
                                        CUBLAS_OP_N,
                                        CUBLAS_OP_N,
                                        n,
                                        n,
                                        n,
                                        &alpha,
                                        d_B,
                                        n,
                                        d_A,
                                        n,
                                        &beta,
                                        d_C,
                                        n);
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        cudaStreamDestroy(stream);
        throw std::runtime_error("cuBLAS SGEMM failed");
    }

    std::vector<float> c_transposed(matrix_size);
    cudaMemcpyAsync(c_transposed.data(), d_C, bytes_needed, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    cudaError_t cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(cuda_err));
    }

    std::vector<float> c(matrix_size);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            c[i * n + j] = c_transposed[j * n + i];
        }
    }
    
    return c;
}

struct AutoCleanup {
    ~AutoCleanup() {
        cleanup();
    }
};

static AutoCleanup auto_cleanup;