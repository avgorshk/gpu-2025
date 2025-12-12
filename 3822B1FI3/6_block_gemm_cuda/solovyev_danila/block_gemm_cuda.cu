
#include <cuda_runtime.h>
#include <algorithm>

static const int BLOCK_SIZE = 32;

__global__ void BlockGemmKernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* C,
                                int n) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int global_i = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int global_j = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float sum = 0.0f;

    int tiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int t = 0; t < tiles; t++) {

        int a_i = global_i;
        int a_k = t * BLOCK_SIZE + threadIdx.x;

        int b_k = t * BLOCK_SIZE + threadIdx.y;
        int b_j = global_j;

        As[threadIdx.y][threadIdx.x] =
            (a_i < n && a_k < n) ? A[a_i * n + a_k] : 0.0f;

        Bs[threadIdx.y][threadIdx.x] =
            (b_k < n && b_j < n) ? B[b_k * n + b_j] : 0.0f;

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (global_i < n && global_j < n)
        C[global_i * n + global_j] = sum;
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    size_t bytes = size_t(n) * n * sizeof(float);
    float *dA, *dB, *dC;
    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);
    cudaMemcpy(dA, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, b.data(), bytes, cudaMemcpyHostToDevice);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    BlockGemmKernel<<<grid, block>>>(dA, dB, dC, n);
    cudaDeviceSynchronize();
    std::vector<float> c(n * n);
    cudaMemcpy(c.data(), dC, bytes, cudaMemcpyDeviceToHost);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return c;
}
