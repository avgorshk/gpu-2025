#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <stdexcept>

constexpr int TILE_DIM = 32;
constexpr int BLOCK_ROWS = 8;

__global__ void block_matrix_multiply_kernel(const float* __restrict__ matrix_a,
    const float* __restrict__ matrix_b,
    float* __restrict__ result_matrix,
    int matrix_size) {
    __shared__ float shared_a[TILE_DIM][TILE_DIM];
    __shared__ float shared_b[TILE_DIM][TILE_DIM];

    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;

    int row = block_row * TILE_DIM + thread_row;
    int col = block_col * TILE_DIM + thread_col;

    float sum = 0.0f;

    int num_blocks = (matrix_size + TILE_DIM - 1) / TILE_DIM;

    for (int block_index = 0; block_index < num_blocks; ++block_index) {
        int a_col = block_index * TILE_DIM + thread_col;
        int a_row = block_row * TILE_DIM + thread_row;
        if (a_row < matrix_size && a_col < matrix_size) {
            shared_a[thread_row][thread_col] = matrix_a[a_row * matrix_size + a_col];
        }
        else {
            shared_a[thread_row][thread_col] = 0.0f;
        }

        int b_row = block_index * TILE_DIM + thread_row;
        int b_col = block_col * TILE_DIM + thread_col;
        if (b_row < matrix_size && b_col < matrix_size) {
            shared_b[thread_row][thread_col] = matrix_b[b_row * matrix_size + b_col];
        }
        else {
            shared_b[thread_row][thread_col] = 0.0f;
        }

        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            sum += shared_a[thread_row][k] * shared_b[k][thread_col];
        }

        __syncthreads();
    }

    if (row < matrix_size && col < matrix_size) {
        result_matrix[row * matrix_size + col] = sum;
    }
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {
    if (a.size() != static_cast<size_t>(n * n) || b.size() != static_cast<size_t>(n * n)) {
        throw std::invalid_argument("Matrix sizes do not match");
    }

    std::vector<float> c(n * n, 0.0f);
    if (n == 0) return c;

    float* d_a, * d_b, * d_c;
    size_t size_bytes = n * n * sizeof(float);

    checkCudaError(cudaMalloc(&d_a, size_bytes), "Allocate A");
    checkCudaError(cudaMalloc(&d_b, size_bytes), "Allocate B");
    checkCudaError(cudaMalloc(&d_c, size_bytes), "Allocate C");

    cudaStream_t stream;
    checkCudaError(cudaStreamCreate(&stream), "Create stream");

    checkCudaError(cudaMemcpyAsync(d_a, a.data(), size_bytes, cudaMemcpyHostToDevice, stream), "Copy A to device");
    checkCudaError(cudaMemcpyAsync(d_b, b.data(), size_bytes, cudaMemcpyHostToDevice, stream), "Copy B to device");

    dim3 block_size(TILE_DIM, TILE_DIM);
    dim3 grid_size((n + TILE_DIM - 1) / TILE_DIM, (n + TILE_DIM - 1) / TILE_DIM);

    block_matrix_multiply_kernel << <grid_size, block_size, 0, stream >> > (d_a, d_b, d_c, n);
    checkCudaError(cudaGetLastError(), "Kernel execution");

    checkCudaError(cudaMemcpyAsync(c.data(), d_c, size_bytes, cudaMemcpyDeviceToHost, stream), "Copy result to host");

    checkCudaError(cudaStreamSynchronize(stream), "Stream sync");

    cudaStreamDestroy(stream);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}