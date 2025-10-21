#include "fft_cufft.h"
#include <cufft.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <vector>

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    // Проверка входных данных
    if (input.size() % (2 * batch) != 0) {
        throw std::invalid_argument("Input size must be divisible by 2 * batch");
    }
    
    int n = input.size() / (2 * batch);  // Длина каждого сигнала в комплексных числах
    
    if (n == 0) {
        throw std::invalid_argument("Signal length cannot be zero");
    }

    cufftResult cufftStatus;
    cudaError_t cudaStatus;
    
    // Создание плана для прямого БПФ
    cufftHandle planForward;
    cufftStatus = cufftPlan1d(&planForward, n, CUFFT_C2C, batch);
    if (cufftStatus != CUFFT_SUCCESS) {
        throw std::runtime_error("Failed to create forward FFT plan");
    }
    
    // Создание плана для обратного БПФ
    cufftHandle planInverse;
    cufftStatus = cufftPlan1d(&planInverse, n, CUFFT_C2C, batch);
    if (cufftStatus != CUFFT_SUCCESS) {
        cufftDestroy(planForward);
        throw std::runtime_error("Failed to create inverse FFT plan");
    }

    // Выделение памяти на устройстве
    cufftComplex *d_input, *d_fft, *d_result;
    size_t complexSize = n * batch * sizeof(cufftComplex);
    
    cudaStatus = cudaMalloc(&d_input, complexSize);
    if (cudaStatus != cudaSuccess) {
        cufftDestroy(planForward);
        cufftDestroy(planInverse);
        throw std::runtime_error("CUDA memory allocation for input failed");
    }
    
    cudaStatus = cudaMalloc(&d_fft, complexSize);
    if (cudaStatus != cudaSuccess) {
        cudaFree(d_input);
        cufftDestroy(planForward);
        cufftDestroy(planInverse);
        throw std::runtime_error("CUDA memory allocation for FFT failed");
    }
    
    cudaStatus = cudaMalloc(&d_result, complexSize);
    if (cudaStatus != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_fft);
        cufftDestroy(planForward);
        cufftDestroy(planInverse);
        throw std::runtime_error("CUDA memory allocation for result failed");
    }

    try {
        // Копирование входных данных на устройство
        // Преобразование из пар float в cufftComplex
        cudaStatus = cudaMemcpy(d_input, input.data(), complexSize, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("Failed to copy input data to device");
        }
        
        // Прямое преобразование Фурье
        cufftStatus = cufftExecC2C(planForward, d_input, d_fft, CUFFT_FORWARD);
        if (cufftStatus != CUFFT_SUCCESS) {
            throw std::runtime_error("Forward FFT execution failed");
        }
        
        // Обратное преобразование Фурье
        cufftStatus = cufftExecC2C(planInverse, d_fft, d_result, CUFFT_INVERSE);
        if (cufftStatus != CUFFT_SUCCESS) {
            throw std::runtime_error("Inverse FFT execution failed");
        }
        
        // Нормализация результата на устройстве
        float scale = 1.0f / n;
        
        // Запускаем ядро для нормализации
        int totalElements = n * batch;
        int blockSize = 256;
        int numBlocks = (totalElements + blockSize - 1) / blockSize;
        
        // Kernel для нормализации
        auto normalizeKernel = [](cufftComplex* data, float scale, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                data[idx].x *= scale;
                data[idx].y *= scale;
            }
        };
        
        normalizeKernel<<<numBlocks, blockSize>>>(d_result, scale, totalElements);
        
        // Проверка ошибок ядра
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("Normalization kernel failed");
        }
        
        // Синхронизация для обеспечения завершения всех операций
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("CUDA device synchronization failed");
        }
        
        // Копирование результата обратно на хост
        std::vector<float> output(input.size());
        cudaStatus = cudaMemcpy(output.data(), d_result, complexSize, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("Failed to copy result data from device");
        }
        
        // Освобождение ресурсов
        cudaFree(d_input);
        cudaFree(d_fft);
        cudaFree(d_result);
        cufftDestroy(planForward);
        cufftDestroy(planInverse);
        
        return output;

    } catch (const std::exception& e) {
        // Освобождение ресурсов в случае ошибки
        cudaFree(d_input);
        cudaFree(d_fft);
        cudaFree(d_result);
        cufftDestroy(planForward);
        cufftDestroy(planInverse);
        throw;
    }
}