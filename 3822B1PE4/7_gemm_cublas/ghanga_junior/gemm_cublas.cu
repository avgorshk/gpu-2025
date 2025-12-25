#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    // Проверка входных данных
    if (a.size() != n * n || b.size() != n * n) {
        throw std::invalid_argument("Input matrices must have size n*n");
    }

    cublasStatus_t status;
    cudaError_t cudaStatus;
    
    // Создание handle для cuBLAS
    cublasHandle_t handle;
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS handle creation failed");
    }

    // Выделение памяти на устройстве
    float *d_A, *d_B, *d_C;
    size_t size = n * n * sizeof(float);
    
    cudaStatus = cudaMalloc(&d_A, size);
    if (cudaStatus != cudaSuccess) {
        cublasDestroy(handle);
        throw std::runtime_error("CUDA memory allocation for A failed");
    }
    
    cudaStatus = cudaMalloc(&d_B, size);
    if (cudaStatus != cudaSuccess) {
        cudaFree(d_A);
        cublasDestroy(handle);
        throw std::runtime_error("CUDA memory allocation for B failed");
    }
    
    cudaStatus = cudaMalloc(&d_C, size);
    if (cudaStatus != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        cublasDestroy(handle);
        throw std::runtime_error("CUDA memory allocation for C failed");
    }

    try {
        // Копирование матриц A и B на устройство
        // Поскольку наши матрицы хранятся по строкам, а cuBLAS ожидает по столбцам,
        // мы будем использовать транспонированные версии в вычислениях
        status = cublasSetMatrix(n, n, sizeof(float), a.data(), n, d_A, n);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to set matrix A on device");
        }
        
        status = cublasSetMatrix(n, n, sizeof(float), b.data(), n, d_B, n);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to set matrix B on device");
        }

        // Параметры для умножения матриц
        const float alpha = 1.0f;
        const float beta = 0.0f;
        
        // Выполнение умножения матриц: C = A × B
        // Поскольку cuBLAS ожидает матрицы по столбцам, а у нас по строкам,
        // то A_row-major × B_row-major = (A_col-major^T × B_col-major^T)^T
        // Поэтому вычисляем: C_col-major = B_col-major × A_col-major
        // Что эквивалентно: C_row-major = A_row-major × B_row-major
        status = cublasSgemm(handle,
                            CUBLAS_OP_N,    // нет транспонирования для B (уже в col-major)
                            CUBLAS_OP_N,    // нет транспонирования для A (уже в col-major)
                            n, n, n,        // размеры матриц
                            &alpha,         // alpha
                            d_B, n,         // B в col-major (эквивалентно b в row-major)
                            d_A, n,         // A в col-major (эквивалентно a в row-major)
                            &beta,          // beta
                            d_C, n);        // C в col-major
        
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cuBLAS SGEMM operation failed");
        }

        // Синхронизация для обеспечения завершения вычислений
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("CUDA device synchronization failed");
        }

        // Копирование результата обратно на хост
        std::vector<float> c(n * n);
        status = cublasGetMatrix(n, n, sizeof(float), d_C, n, c.data(), n);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to get result matrix from device");
        }

        // Освобождение ресурсов
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cublasDestroy(handle);

        return c;

    } catch (const std::exception& e) {
        // Освобождение ресурсов в случае ошибки
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cublasDestroy(handle);
        throw;
    }
}