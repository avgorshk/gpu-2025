#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <iostream>

__global__ void kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int n) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * (blockDim.x * 4) + threadIdx.x;
    
    if (x >= n) return;
    
    float sum[4] = {0};
    int cols[4];
    
    for (int i = 0; i < 4; i++) {
        cols[i] = y + i * blockDim.x;
        if (cols[i] >= n) continue;
        
        for (int k = 0; k < n; k++) {
            sum[i] += a[x * n + k] * b[k * n + cols[i]];
        }
    }
    
    for (int i = 0; i < 4; i++) {
        if (cols[i] < n) {
            c[x * n + cols[i]] = sum[i];
        }
    }
}

std::vector<float> NaiveGemmCUDA(
    const std::vector<float>& a,
    const std::vector<float>& b,
    int n
) {
    size_t size = n * n * sizeof(float);
    float *dev_a, *dev_b, *dev_c;
    std::vector<float> c(n * n, 0);

    cudaMalloc(&dev_a, size);
    cudaMalloc(&dev_b, size);
    cudaMalloc(&dev_c, size);

    cudaMemcpy(dev_a, a.data(), size, cudaMemcpyHostToDevice);
    
    dim3 block_dim(32, 8);
    
    int elems_per_block = block_dim.x * 4;
    dim3 num_blocks((n + elems_per_block - 1) / elems_per_block, 
                    (n + block_dim.y - 1) / block_dim.y);
    cudaMemcpy(dev_b, b.data(), size, cudaMemcpyHostToDevice);

    kernel<<<num_blocks, block_dim>>>(dev_a, dev_b, dev_c, n);

    cudaMemcpy(c.data(), dev_c, size, cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return c;
}