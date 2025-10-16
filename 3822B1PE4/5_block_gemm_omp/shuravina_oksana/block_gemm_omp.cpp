#include "block_gemm_omp.h"
#include <vector>
#include <omp.h>
#include <cmath>
#include <iostream>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(n * n, 0.0f);
    
    // Проверка что размер матрицы - степень двойки
    if ((n & (n - 1)) != 0) {
        std::cerr << "Warning: Matrix size is not power of 2" << std::endl;
    }
    
    // Определяем количество блоков
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Основной цикл по блокам с параллелизацией
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int block_i = 0; block_i < num_blocks; ++block_i) {
        for (int block_j = 0; block_j < num_blocks; ++block_j) {
            
            // Границы текущего блока в матрице C
            int i_start = block_i * BLOCK_SIZE;
            int i_end = std::min(i_start + BLOCK_SIZE, n);
            
            int j_start = block_j * BLOCK_SIZE;
            int j_end = std::min(j_start + BLOCK_SIZE, n);
            
            // Временный буфер для блока результата (улучшает локальность)
            float block_c[BLOCK_SIZE][BLOCK_SIZE] = {0};
            
            // Цикл по внутренним блокам (k-блоки)
            for (int block_k = 0; block_k < num_blocks; ++block_k) {
                int k_start = block_k * BLOCK_SIZE;
                int k_end = std::min(k_start + BLOCK_SIZE, n);
                
                // Умножение текущих блоков A и B
                for (int i = i_start; i < i_end; ++i) {
                    int local_i = i - i_start;
                    
                    for (int k = k_start; k < k_end; ++k) {
                        float a_val = a[i * n + k];
                        
                        // Векторизованный внутренний цикл
                        #pragma omp simd
                        for (int j = j_start; j < j_end; ++j) {
                            int local_j = j - j_start;
                            block_c[local_i][local_j] += a_val * b[k * n + j];
                        }
                    }
                }
            }
            
            // Запись блока результата в итоговую матрицу
            for (int i = i_start; i < i_end; ++i) {
                int local_i = i - i_start;
                #pragma omp simd
                for (int j = j_start; j < j_end; ++j) {
                    int local_j = j - j_start;
                    c[i * n + j] = block_c[local_i][local_j];
                }
            }
        }
    }
    
    return c;
}