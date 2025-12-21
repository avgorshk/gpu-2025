#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>

constexpr int TILE_SIZE = 32;

__global__ void GemmKernel(const float* A, const float* B, float* C, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int globalRow = blockIdx.y * TILE_SIZE + threadIdx.y;
    int globalCol = blockIdx.x * TILE_SIZE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < N; t += TILE_SIZE) {
        tileA[threadIdx.y][threadIdx.x] = (globalRow < N && t + threadIdx.x < N)
            ? A[globalRow * N + t + threadIdx.x] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] = (globalCol < N && t + threadIdx.y < N)
            ? B[(t + threadIdx.y) * N + globalCol] : 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i)
            acc += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];

        __syncthreads();
    }

    if (globalRow < N && globalCol < N)
        C[globalRow * N + globalCol] = acc;
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& A,
                                 const std::vector<float>& B, int N) {
    size_t size = N * N * sizeof(float);
    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), size, cudaMemcpyHostToDevice);

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                 (N + TILE_SIZE - 1) / TILE_SIZE);

    GemmKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);

    std::vector<float> C(N * N);
    cudaMemcpy(C.data(), d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return C;
}