#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <iostream>
#include <stdexcept>

std::vector<float> GemmCUBLAS(const std::vector<float> &a,
                              const std::vector<float> &b,
                              int n)
{
    std::vector<float> c(n * n, 0.0f);

    if (a.size() != n * n || b.size() != n * n)
    {
        throw std::invalid_argument("Неверный размер входных матриц");
    }

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

    size_t matrix_size = n * n * sizeof(float);
    cudaError_t cuda_err;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cuda_err = cudaMallocAsync(&d_a, matrix_size, stream);
    if (cuda_err != cudaSuccess)
    {
        throw std::runtime_error("Ошибка выделения памяти для матрицы A: " +
                                 std::string(cudaGetErrorString(cuda_err)));
    }

    cuda_err = cudaMallocAsync(&d_b, matrix_size, stream);
    if (cuda_err != cudaSuccess)
    {
        cudaFreeAsync(d_a, stream);
        throw std::runtime_error("Ошибка выделения памяти для матрицы B: " +
                                 std::string(cudaGetErrorString(cuda_err)));
    }

    cuda_err = cudaMallocAsync(&d_c, matrix_size, stream);
    if (cuda_err != cudaSuccess)
    {
        cudaFreeAsync(d_a, stream);
        cudaFreeAsync(d_b, stream);
        throw std::runtime_error("Ошибка выделения памяти для матрицы C: " +
                                 std::string(cudaGetErrorString(cuda_err)));
    }

    cuda_err = cudaMemcpyAsync(d_a, a.data(), matrix_size, cudaMemcpyHostToDevice, stream);
    if (cuda_err != cudaSuccess)
    {
        cleanup(d_a, d_b, d_c, stream, nullptr);
        throw std::runtime_error("Ошибка копирования матрицы A: " +
                                 std::string(cudaGetErrorString(cuda_err)));
    }

    cuda_err = cudaMemcpyAsync(d_b, b.data(), matrix_size, cudaMemcpyHostToDevice, stream);
    if (cuda_err != cudaSuccess)
    {
        cleanup(d_a, d_b, d_c, stream, nullptr);
        throw std::runtime_error("Ошибка копирования матрицы B: " +
                                 std::string(cudaGetErrorString(cuda_err)));
    }

    cublasHandle_t handle;
    cublasStatus_t cublas_err = cublasCreate(&handle);
    if (cublas_err != CUBLAS_STATUS_SUCCESS)
    {
        cleanup(d_a, d_b, d_c, stream, nullptr);
        throw std::runtime_error("Ошибка создания handle cuBLAS");
    }

    cublasSetStream(handle, stream);

    float alpha = 1.0f;
    float beta = 0.0f;

    cublas_err = cublasSgemm(handle,
                             CUBLAS_OP_N,
                             CUBLAS_OP_N,
                             n, n, n,
                             &alpha,
                             d_a, n,
                             d_b, n,
                             &beta,
                             d_c, n);

    if (cublas_err != CUBLAS_STATUS_SUCCESS)
    {
        cleanup(d_a, d_b, d_c, stream, handle);
        throw std::runtime_error("Ошибка умножения матриц cuBLAS");
    }

    cuda_err = cudaMemcpyAsync(c.data(), d_c, matrix_size, cudaMemcpyDeviceToHost, stream);
    if (cuda_err != cudaSuccess)
    {
        cleanup(d_a, d_b, d_c, stream, handle);
        throw std::runtime_error("Ошибка копирования результата: " +
                                 std::string(cudaGetErrorString(cuda_err)));
    }

    cudaStreamSynchronize(stream);

    cleanup(d_a, d_b, d_c, stream, handle);

    return c;
}

void cleanup(float *d_a, float *d_b, float *d_c, cudaStream_t stream, cublasHandle_t handle)
{
    if (d_a)
        cudaFreeAsync(d_a, stream);
    if (d_b)
        cudaFreeAsync(d_b, stream);
    if (d_c)
        cudaFreeAsync(d_c, stream);
    if (stream)
        cudaStreamDestroy(stream);
    if (handle)
        cublasDestroy(handle);
}