#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

#define TILE_SIZE 16
#define BLOCK_SIZE 256

// Базовое ядро для проверки правильности
__global__ void naive_gemm_basic_kernel(const float* a, const float* b, float* c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

// Оптимизированное ядро с тайлингом и shared memory
__global__ void naive_gemm_optimized_kernel(const float* a, const float* b, float* c, int n) {
    __shared__ float tile_a[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_b[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int k = 0; k < n; k += TILE_SIZE) {
        // Загрузка тайлов в shared memory
        if (row < n && (k + tx) < n) {
            tile_a[ty][tx] = a[row * n + k + tx];
        } else {
            tile_a[ty][tx] = 0.0f;
        }
        
        if ((k + ty) < n && col < n) {
            tile_b[ty][tx] = b[(k + ty) * n + col];
        } else {
            tile_b[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Вычисления с использованием shared memory
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += tile_a[ty][i] * tile_b[i][tx];
        }
        
        __syncthreads();
    }
    
    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}

// Ядро с векторизованной загрузкой
__global__ void naive_gemm_vectorized_kernel(const float* a, const float* b, float* c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        
        // Векторизованная загрузка по 4 элемента
        for (int k = 0; k < n; k += 4) {
            float4 a_vec, b_vec;
            
            // Загрузка 4 элементов из a
            if (k + 3 < n) {
                a_vec = *reinterpret_cast<const float4*>(&a[row * n + k]);
            } else {
                // Обработка граничных случаев
                a_vec.x = a[row * n + k];
                a_vec.y = (k + 1 < n) ? a[row * n + k + 1] : 0.0f;
                a_vec.z = (k + 2 < n) ? a[row * n + k + 2] : 0.0f;
                a_vec.w = (k + 3 < n) ? a[row * n + k + 3] : 0.0f;
            }
            
            // Загрузка 4 элементов из b (транспонированный доступ)
            if (k + 3 < n) {
                b_vec.x = b[k * n + col];
                b_vec.y = b[(k + 1) * n + col];
                b_vec.z = b[(k + 2) * n + col];
                b_vec.w = b[(k + 3) * n + col];
            } else {
                b_vec.x = b[k * n + col];
                b_vec.y = (k + 1 < n) ? b[(k + 1) * n + col] : 0.0f;
                b_vec.z = (k + 2 < n) ? b[(k + 2) * n + col] : 0.0f;
                b_vec.w = (k + 3 < n) ? b[(k + 3) * n + col] : 0.0f;
            }
            
            sum.x += a_vec.x * b_vec.x;
            sum.y += a_vec.y * b_vec.y;
            sum.z += a_vec.z * b_vec.z;
            sum.w += a_vec.w * b_vec.w;
        }
        
        c[row * n + col] = sum.x + sum.y + sum.z + sum.w;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    // Проверка входных данных
    if (a.size() != n * n || b.size() != n * n) {
        std::cerr << "Error: Invalid matrix dimensions" << std::endl;
        return std::vector<float>();
    }
    
    // Проверка что n - степень двойки
    if ((n & (n - 1)) != 0) {
        std::cerr << "Warning: Matrix size is not power of 2, performance may be suboptimal" << std::endl;
    }
    
    // Выделение памяти на устройстве
    float *d_a, *d_b, *d_c;
    size_t size = n * n * sizeof(float);
    
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    // Асинхронная копия данных на устройство
    cudaMemcpyAsync(d_a, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_b, b.data(), size, cudaMemcpyHostToDevice);
    
    // Выбор оптимальной конфигурации запуска
    dim3 blockDim, gridDim;
    
    if (n <= 32) {
        // Для маленьких матриц используем базовое ядро
        blockDim = dim3(16, 16);
        gridDim = dim3((n + 15) / 16, (n + 15) / 16);
        naive_gemm_basic_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_c, n);
    } else if (n <= 1024) {
        // Для средних матриц используем тайлинг
        blockDim = dim3(TILE_SIZE, TILE_SIZE);
        gridDim = dim3((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);
        naive_gemm_optimized_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_c, n);
    } else {
        // Для больших матриц используем векторизацию
        blockDim = dim3(16, 16);
        gridDim = dim3((n + 15) / 16, (n + 15) / 16);
        naive_gemm_vectorized_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_c, n);
    }
    
    // Проверка ошибок ядра
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return std::vector<float>();
    }
    
    // Синхронизация и копирование результата
    cudaDeviceSynchronize();
    
    std::vector<float> c(n * n);
    cudaMemcpy(c.data(), d_c, size, cudaMemcpyDeviceToHost);
    
    // Освобождение памяти
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return c;
}