#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <cstring>

#define BLOCK_SIZE 32

__global__ void NaiveGemmSharedKernel(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* C,
                                      int N) {
    __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float sum = 0.0f;

    int num_tiles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        int a_col = t * BLOCK_SIZE + threadIdx.x;
        int b_row = t * BLOCK_SIZE + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] = (row < N && a_col < N) ? A[row * N + a_col] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (col < N && b_row < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    size_t bytes = static_cast<size_t>(n) * n * sizeof(float);
    std::vector<float> c(n * n, 0.0f);

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    float *h_a = nullptr, *h_b = nullptr;

    cudaHostAlloc((void**)&h_a, bytes, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_b, bytes, cudaHostAllocDefault);

    std::memcpy(h_a, a.data(), bytes);
    std::memcpy(h_b, b.data(), bytes);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(d_a, h_a, bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, h_b, bytes, cudaMemcpyHostToDevice, stream);

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    NaiveGemmSharedKernel<<<blocks, threads, 0, stream>>>(d_a, d_b, d_c, n);

    cudaMemcpyAsync(c.data(), d_c, bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaStreamDestroy(stream);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);

    return c;
}
