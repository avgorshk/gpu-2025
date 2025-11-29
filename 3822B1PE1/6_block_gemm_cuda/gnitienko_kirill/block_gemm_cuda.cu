#include "block_gemm_cuda.h"
#include <cuda_runtime.h>

template <int BLOCK_SIZE>
__global__ void block_gemm_kernel_template(const float* A, const float* B, float* C, int n) {

    __shared__ float shared_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shared_B[BLOCK_SIZE][BLOCK_SIZE];

    int i_block = blockIdx.y;
    int j_block = blockIdx.x;
    int i_local = threadIdx.y;
    int j_local = threadIdx.x;

    int i_global = i_block * BLOCK_SIZE + i_local;
    int j_global = j_block * BLOCK_SIZE + j_local;

    float sum = 0.0f;
    int num_blocks = n / BLOCK_SIZE;

    for (int k_block = 0; k_block < num_blocks; ++k_block) {

        shared_A[i_local][j_local] = A[i_global * n + k_block * BLOCK_SIZE + j_local];
        shared_B[i_local][j_local] = B[(k_block * BLOCK_SIZE + i_local) * n + j_global];

        __syncthreads();

#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += shared_A[i_local][k] * shared_B[k][j_local];
        }

        __syncthreads();
    }

    C[i_global * n + j_global] = sum;
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) {

    int block_size;
    if (n <= 64) {
        block_size = 8;
    }
    else if (n <= 256) {
        block_size = 16;
    }
    else if (n <= 1024) {
        block_size = 32;
    }
    else {
        block_size = 64;
    }

    const int size = n * n;
    float* data_A, * data_B, * data_C;
    cudaMalloc(&data_A, size * sizeof(float));
    cudaMalloc(&data_B, size * sizeof(float));
    cudaMalloc(&data_C, size * sizeof(float));

    cudaMemcpy(data_A, a.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(data_B, b.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(block_size, block_size);
    dim3 blocks((n + block_size - 1) / block_size, (n + block_size - 1) / block_size);

    switch (block_size) {
        case 8:
            block_gemm_kernel_template<8><<<blocks, threads>>>(data_A, data_B, data_C, n);
            break;
        case 16:
            block_gemm_kernel_template<16><<<blocks, threads>>>(data_A, data_B, data_C, n);
            break;
        case 32:
            block_gemm_kernel_template<32><<<blocks, threads>>>(data_A, data_B, data_C, n);
            break;
        case 64:
            block_gemm_kernel_template<64><<<blocks, threads>>>(data_A, data_B, data_C, n);
            break;
    }
    cudaDeviceSynchronize();

    std::vector<float> c(n * n);
    cudaMemcpy(c.data(), data_C, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(data_A);
    cudaFree(data_B);
    cudaFree(data_C);

    return c;
}