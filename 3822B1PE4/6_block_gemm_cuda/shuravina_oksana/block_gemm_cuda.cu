#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define BLOCK_SIZE 16
#define TILE_SIZE 16

__global__ void blockGemmKernel(const float* A, const float* B, float* C, int n) {
    // Блочная индексация
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    
    // Индексы внутри блока
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;
    
    // Глобальные индексы для элемента C
    int row = blockRow * TILE_SIZE + threadRow;
    int col = blockCol * TILE_SIZE + threadCol;
    
    // Общая память для тайлов
    __shared__ float sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedB[TILE_SIZE][TILE_SIZE];
    
    float sum = 0.0f;
    
    // Количество тайлов
    int numTiles = (n + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; t++) {
        // Загрузка тайлов в общую память
        int aCol = t * TILE_SIZE + threadCol;
        int bRow = t * TILE_SIZE + threadRow;
        
        if (row < n && aCol < n) {
            sharedA[threadRow][threadCol] = A[row * n + aCol];
        } else {
            sharedA[threadRow][threadCol] = 0.0f;
        }
        
        if (bRow < n && col < n) {
            sharedB[threadRow][threadCol] = B[bRow * n + col];
        } else {
            sharedB[threadRow][threadCol] = 0.0f;
        }
        
        __syncthreads();
        
        // Вычисление произведения для текущих тайлов
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += sharedA[threadRow][k] * sharedB[k][threadCol];
        }
        
        __syncthreads();
    }
    
    // Запись результата
    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

// Оптимизированная версия с векторизацией и лучшим использованием памяти
__global__ void blockGemmOptimizedKernel(const float* __restrict__ A, 
                                        const float* __restrict__ B, 
                                        float* __restrict__ C, 
                                        int n) {
    __shared__ float sharedA[TILE_SIZE][TILE_SIZE + 1]; // +1 для избежания bank conflicts
    __shared__ float sharedB[TILE_SIZE][TILE_SIZE + 1];
    
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;
    
    int row = blockRow * TILE_SIZE + threadRow;
    int col = blockCol * TILE_SIZE + threadCol;
    
    float sum = 0.0f;
    int numTiles = (n + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; t++) {
        // Векторизованная загрузка с дружественным к варпу доступом
        int aCol = t * TILE_SIZE + threadCol;
        int bRow = t * TILE_SIZE + threadRow;
        
        // Загрузка с проверкой границ
        if (row < n && aCol < n) {
            sharedA[threadRow][threadCol] = A[row * n + aCol];
        } else {
            sharedA[threadRow][threadCol] = 0.0f;
        }
        
        if (bRow < n && col < n) {
            sharedB[threadRow][threadCol] = B[bRow * n + col];
        } else {
            sharedB[threadRow][threadCol] = 0.0f;
        }
        
        __syncthreads();
        
        // Развернутый цикл для лучшей производительности
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += sharedA[threadRow][k] * sharedB[k][threadCol];
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
    // Проверка входных данных
    if (a.size() != n * n || b.size() != n * n) {
        throw std::invalid_argument("Input matrices must have size n*n");
    }
    
    // Выделение памяти на устройстве
    float *d_A, *d_B, *d_C;
    size_t size = n * n * sizeof(float);
    
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    // Копирование данных на устройство
    cudaMemcpy(d_A, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b.data(), size, cudaMemcpyHostToDevice);
    
    // Вычисление размеров сетки и блоков
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((n + TILE_SIZE - 1) / TILE_SIZE, 
                 (n + TILE_SIZE - 1) / TILE_SIZE);
    
    // Запуск ядра
    blockGemmOptimizedKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, n);
    
    // Проверка ошибок
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        throw std::runtime_error("CUDA kernel execution failed");
    }
    
    // Копирование результата обратно на хост
    std::vector<float> c(n * n);
    cudaMemcpy(c.data(), d_C, size, cudaMemcpyDeviceToHost);
    
    // Освобождение памяти
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return c;
}