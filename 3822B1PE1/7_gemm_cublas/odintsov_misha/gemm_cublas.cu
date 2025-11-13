#include "gemm_cublas.h"
#include <vector>
#include <cstring> 

// CUDA и CUBLAS
#include <cuda_runtime.h>
#include <cublas_v2.h>



std::vector<float> GemmCUBLAS(const std::vector<float>& a, const std::vector<float>& b, int n) {
    size_t len = n * n;
    std::vector<float> output(len, 0.0f);

    // Выделение памяти на устройстве
    float* d_a;
    float* d_b;
    float* d_c;

    cudaMalloc((void**)&d_a, len * sizeof(float));
    cudaMalloc((void**)&d_b, len * sizeof(float));
    cudaMalloc((void**)&d_c, len * sizeof(float));

    // Использование фиксированной памяти на хосте
    float* h_a;
    float* h_b;
    cudaMallocHost((void**)&h_a, len * sizeof(float));
    cudaMallocHost((void**)&h_b, len * sizeof(float));

    // Копирование данных из std::vector в зафиксированную память
    std::memcpy(h_a, a.data(), len * sizeof(float));
    std::memcpy(h_b, b.data(), len * sizeof(float));

    // Копирование данных в память устройства
    cudaMemcpy(d_a, h_a, len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, len * sizeof(float), cudaMemcpyHostToDevice);

    // Инициализация CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Выполнение матричного умножения
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, n, &alpha, d_a, n, d_b, n, &beta, d_c, n);

    // Копирование результата обратно на хост
    cudaMemcpy(output.data(), d_c, len * sizeof(float), cudaMemcpyDeviceToHost);

    // Освобождение памяти
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cublasDestroy(handle);

    return output;
}