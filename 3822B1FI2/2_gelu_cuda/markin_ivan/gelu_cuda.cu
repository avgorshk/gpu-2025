#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>

__device__ float gelu_elementwise(float x) {
    const float t = 0.7978845608028654f * x + 0.035677f * x * x * x; // предвычислено 
    return 0.5f * x * (1.0f + t * (27.0f + t*t) / (27.0f + 9.0f * t*t));
}

__global__ void gelu_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x * 4;  // Каждый поток обрабатывает 4 элемента
    
    // Обрабатываем по 4 элемента на поток
    for (int i = idx * 4; i < n; i += stride) {
        // Обрабатываем 4 элемента последовательно
        if (i < n) output[i] = gelu_elementwise(input[i]);
        if (i + 1 < n) output[i + 1] = gelu_elementwise(input[i + 1]);
        if (i + 2 < n) output[i + 2] = gelu_elementwise(input[i + 2]);
        if (i + 3 < n) output[i + 3] = gelu_elementwise(input[i + 3]);
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const size_t n = input.size();
    std::vector<float> result(n);
    
    // Выделяем память на GPU
    float *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));
    
    // Копируем данные на GPU
    cudaMemcpy(d_input, input.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Конфигурация запуска ядра
    const int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;
    
    // Запускаем ядро
    gelu_kernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    
    // Копируем результат обратно на CPU
    cudaMemcpy(result.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Освобождаем память GPU
    cudaFree(d_input);
    cudaFree(d_output);
    
    return result;
}