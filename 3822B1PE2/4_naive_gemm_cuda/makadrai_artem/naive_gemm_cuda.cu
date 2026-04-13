#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>

#define BLOCK_SIZE 16

__global__ void naive_gemm_kernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ c,
                                  int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        float sum = 0.0f;
        
        const float4* a_vec = reinterpret_cast<const float4*>(a);
        const int n_vec = n / 4;
        const int row_vec = row * n_vec;
        
        int k = 0;
        for (; k < n_vec; ++k) {
            float4 a_vals = a_vec[row_vec + k];
            int k_base = k * 4;
            sum += a_vals.x * b[(k_base + 0) * n + col];
            sum += a_vals.y * b[(k_base + 1) * n + col];
            sum += a_vals.z * b[(k_base + 2) * n + col];
            sum += a_vals.w * b[(k_base + 3) * n + col];
        }
        
        for (int k_idx = n_vec * 4; k_idx < n; ++k_idx) {
            sum += a[row * n + k_idx] * b[k_idx * n + col];
        }
        
        c[row * n + col] = sum;
    }
}

static float* d_a = nullptr;
static float* d_b = nullptr;
static float* d_c = nullptr;
static cudaStream_t stream = nullptr;
static size_t allocated_size = 0;
static bool initialized = false;

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n <= 0) {
        return std::vector<float>();
    }
    
    size_t matrix_size = static_cast<size_t>(n) * n;
    size_t bytes = matrix_size * sizeof(float);
    
    if (!initialized) {
        cudaStreamCreate(&stream);
        initialized = true;
    }
    
    if (d_a == nullptr || allocated_size < matrix_size) {
        if (d_a != nullptr) {
            cudaFree(d_a);
            cudaFree(d_b);
            cudaFree(d_c);
        }
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_c, bytes);
        allocated_size = matrix_size;
    }
    
    cudaMemcpyAsync(d_a, a.data(), bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, b.data(), bytes, cudaMemcpyHostToDevice, stream);
    
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x,
                  (n + blockSize.y - 1) / blockSize.y);
    
    naive_gemm_kernel<<<gridSize, blockSize, 0, stream>>>(d_a, d_b, d_c, n);
    
    std::vector<float> c(matrix_size);
    
    cudaMemcpyAsync(c.data(), d_c, bytes, cudaMemcpyDeviceToHost, stream);
    
    cudaStreamSynchronize(stream);
    
    return c;
}

