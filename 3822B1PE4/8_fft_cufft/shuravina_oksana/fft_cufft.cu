#include "fft_cufft.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>
#include <stdexcept>

__global__ void normalize_kernel(cufftComplex* data, int total, float inv_n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        data[i].x *= inv_n;
        data[i].y *= inv_n;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    // Проверка входных параметров
    if (batch <= 0) {
        throw std::invalid_argument("Batch must be positive");
    }
    if (input.empty()) {
        throw std::invalid_argument("Input cannot be empty");
    }
    if (input.size() % (2 * batch) != 0) {
        throw std::invalid_argument("Input size must be divisible by 2 * batch");
    }

    const int n = input.size() / (2 * batch);  // Длина каждого сигнала в комплексных числах
    const int total_complex = n * batch;
    const size_t bytes = sizeof(cufftComplex) * total_complex;

    // Выделение памяти на устройстве
    cufftComplex* d_data = nullptr;
    cudaError_t cudaStatus = cudaMalloc(&d_data, bytes);
    if (cudaStatus != cudaSuccess) {
        throw std::runtime_error("CUDA memory allocation failed");
    }

    // Копирование данных на устройство
    cudaStatus = cudaMemcpy(d_data, input.data(), bytes, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        cudaFree(d_data);
        throw std::runtime_error("CUDA memcpy to device failed");
    }

    // Создание плана cuFFT
    cufftHandle plan;
    cufftResult cufftStatus = cufftPlan1d(&plan, n, CUFFT_C2C, batch);
    if (cufftStatus != CUFFT_SUCCESS) {
        cudaFree(d_data);
        throw std::runtime_error("cuFFT plan creation failed");
    }

    try {
        // Прямое БПФ
        cufftStatus = cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
        if (cufftStatus != CUFFT_SUCCESS) {
            throw std::runtime_error("Forward FFT failed");
        }

        // Обратное БПФ
        cufftStatus = cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);
        if (cufftStatus != CUFFT_SUCCESS) {
            throw std::runtime_error("Inverse FFT failed");
        }

        // Нормализация
        const float inv_n = 1.0f / static_cast<float>(n);
        const int blockSize = 256;
        const int gridSize = (total_complex + blockSize - 1) / blockSize;
        
        normalize_kernel<<<gridSize, blockSize>>>(d_data, total_complex, inv_n);
        
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("Kernel launch failed");
        }

        // Синхронизация
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("CUDA synchronization failed");
        }

        // Копирование результата обратно
        std::vector<float> output(input.size());
        cudaStatus = cudaMemcpy(output.data(), d_data, bytes, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("CUDA memcpy to host failed");
        }

        // Освобождение ресурсов
        cufftDestroy(plan);
        cudaFree(d_data);

        return output;

    } catch (const std::exception& e) {
        // Освобождение ресурсов при ошибке
        cufftDestroy(plan);
        cudaFree(d_data);
        throw;
    }
}