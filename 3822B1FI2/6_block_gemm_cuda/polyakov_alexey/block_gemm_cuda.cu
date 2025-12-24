#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <cmath>

constexpr int BLOCK_SIZE = 32;

__global__ void block_gemm_kernel(const float* A, const float* B, float* C, int n) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;

    int row = blockRow * BLOCK_SIZE + threadRow;
    int col = blockCol * BLOCK_SIZE + threadCol;

    float c_val = 0.0f;

    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int k_block = 0; k_block < numBlocks; ++k_block) {
        int k_start = k_block * BLOCK_SIZE;

        int a_row = row;
        int a_col = k_start + threadCol;
        if (a_row < n && a_col < n) {
            As[threadRow][threadCol] = A[a_row * n + a_col];
        } else {
            As[threadRow][threadCol] = 0.0f;
        }

        int b_row = k_start + threadRow;
        int b_col = col;
        if (b_row < n && b_col < n) {
            Bs[threadRow][threadCol] = B[b_row * n + b_col];
        } else {
            Bs[threadRow][threadCol] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            c_val += As[threadRow][k] * Bs[k][threadCol];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = c_val;
    }
}

template<int TILE_SIZE>
__global__ void optimized_block_gemm_kernel(const float* __restrict__ A, 
                                            const float* __restrict__ B, 
                                            float* __restrict__ C, 
                                            int n) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + 1];

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;

    float c_vals[TILE_SIZE][TILE_SIZE] = {{0}};

    int startRow = blockRow * BLOCK_SIZE * TILE_SIZE;
    int startCol = blockCol * BLOCK_SIZE * TILE_SIZE;

    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    for (int k_block = 0; k_block < numBlocks; ++k_block) {
        int k_start = k_block * BLOCK_SIZE;

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            int row = startRow + threadRow * TILE_SIZE + i;
            int col = k_start + threadCol;
            if (row < n && col < n) {
                As[threadRow * TILE_SIZE + i][threadCol] = A[row * n + col];
            } else {
                As[threadRow * TILE_SIZE + i][threadCol] = 0.0f;
            }
        }

        #pragma unroll
        for (int j = 0; j < TILE_SIZE; ++j) {
            int row = k_start + threadRow;
            int col = startCol + threadCol * TILE_SIZE + j;
            if (row < n && col < n) {
                Bs[threadRow][threadCol * TILE_SIZE + j] = B[row * n + col];
            } else {
                Bs[threadRow][threadCol * TILE_SIZE + j] = 0.0f;
            }
        }
        
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            float a_reg[TILE_SIZE];
            float b_reg[TILE_SIZE];
            
            #pragma unroll
            for (int i = 0; i < TILE_SIZE; ++i) {
                a_reg[i] = As[threadRow * TILE_SIZE + i][k];
            }
            
            #pragma unroll
            for (int j = 0; j < TILE_SIZE; ++j) {
                b_reg[j] = Bs[k][threadCol * TILE_SIZE + j];
            }

            #pragma unroll
            for (int i = 0; i < TILE_SIZE; ++i) {
                #pragma unroll
                for (int j = 0; j < TILE_SIZE; ++j) {
                    c_vals[i][j] = fmaf(a_reg[i], b_reg[j], c_vals[i][j]);
                }
            }
        }
        
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TILE_SIZE; ++i) {
        #pragma unroll
        for (int j = 0; j < TILE_SIZE; ++j) {
            int row = startRow + threadRow * TILE_SIZE + i;
            int col = startCol + threadCol * TILE_SIZE + j;
            if (row < n && col < n) {
                C[row * n + col] = c_vals[i][j];
            }
        }
    }
}

static float* d_A = nullptr;
static float* d_B = nullptr;
static float* d_C = nullptr;
static size_t allocated_size = 0;

void free_gpu_memory() {
    if (d_A) cudaFree(d_A);
    if (d_B) cudaFree(d_B);
    if (d_C) cudaFree(d_C);
    d_A = d_B = d_C = nullptr;
    allocated_size = 0;
}

bool allocate_gpu_memory(size_t bytes) {
    cudaError_t err;
    
    err = cudaMalloc(&d_A, bytes);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed for A: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    err = cudaMalloc(&d_B, bytes);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed for B: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        d_A = nullptr;
        return false;
    }
    
    err = cudaMalloc(&d_C, bytes);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed for C: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        d_A = d_B = nullptr;
        return false;
    }
    
    allocated_size = bytes;
    return true;
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n <= 0) {
        return std::vector<float>();
    }
    
    const size_t matrix_size = n * n;
    const size_t bytes_needed = matrix_size * sizeof(float);
    
    if (a.size() != matrix_size || b.size() != matrix_size) {
        throw std::runtime_error("Input matrix size mismatch");
    }
    std::vector<float> c(matrix_size, 0.0f);

    if (allocated_size < bytes_needed) {
        free_gpu_memory();
        if (!allocate_gpu_memory(bytes_needed)) {
            throw std::runtime_error("Failed to allocate GPU memory");
        }
    }
    
    cudaError_t err;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    err = cudaMemcpyAsync(d_A, a.data(), bytes_needed, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        cudaStreamDestroy(stream);
        throw std::runtime_error(std::string("cudaMemcpy A failed: ") + cudaGetErrorString(err));
    }
    
    err = cudaMemcpyAsync(d_B, b.data(), bytes_needed, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        cudaStreamDestroy(stream);
        throw std::runtime_error(std::string("cudaMemcpy B failed: ") + cudaGetErrorString(err));
    }

    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    if (n >= 1024) {
        const int TILE_SIZE = 2; // Каждый поток обрабатывает 2x2 элемента
        dim3 opt_block_dim(BLOCK_SIZE / TILE_SIZE, BLOCK_SIZE / TILE_SIZE);
        dim3 opt_grid_dim((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
                          (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
        
        optimized_block_gemm_kernel<TILE_SIZE><<<opt_grid_dim, opt_block_dim, 0, stream>>>(d_A, d_B, d_C, n);
    } else {
        block_gemm_kernel<<<grid_dim, block_dim, 0, stream>>>(d_A, d_B, d_C, n);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaStreamDestroy(stream);
        throw std::runtime_error(std::string("Kernel launch failed: ") + cudaGetErrorString(err));
    }

    err = cudaMemcpyAsync(c.data(), d_C, bytes_needed, cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        cudaStreamDestroy(stream);
        throw std::runtime_error(std::string("cudaMemcpy C failed: ") + cudaGetErrorString(err));
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        cudaStreamDestroy(stream);
        throw std::runtime_error(std::string("cudaStreamSynchronize failed: ") + cudaGetErrorString(err));
    }

    cudaStreamDestroy(stream);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after execution: " << cudaGetErrorString(err) << std::endl;
    }
    
    return c;
}

struct CUDACleanup {
    ~CUDACleanup() {
        free_gpu_memory();
    }
};

static CUDACleanup cleanup;