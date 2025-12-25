#include "block_gemm_cuda.h"

#include <cuda_runtime.h>
#include <stdexcept>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        throw std::runtime_error(cudaGetErrorString(err));         \
    }                                                              \
} while(0)

constexpr int TILE_SIZE = 16;

__global__ void blockGemmKernel(const float* __restrict__ a,
                                 const float* __restrict__ b,
                                 float* __restrict__ c,
                                 int n) {
    __shared__ float sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedB[TILE_SIZE][TILE_SIZE];

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    int row = blockRow * TILE_SIZE + threadIdx.y;
    int col = blockCol * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int tile = 0; tile < n; tile += TILE_SIZE) {
        // Load tile of A into shared memory
        int aCol = tile + threadIdx.x;
        if (row < n && aCol < n) {
            sharedA[threadIdx.y][threadIdx.x] = a[row * n + aCol];
        } else {
            sharedA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile of B into shared memory
        int bRow = tile + threadIdx.y;
        if (bRow < n && col < n) {
            sharedB[threadIdx.y][threadIdx.x] = b[bRow * n + col];
        } else {
            sharedB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += sharedA[threadIdx.y][k] * sharedB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n <= 0) {
        return {};
    }

    const size_t matrixSize = static_cast<size_t>(n) * n;
    if (a.size() != matrixSize || b.size() != matrixSize) {
        throw std::invalid_argument("Matrix sizes do not match n*n");
    }

    const size_t bytes = matrixSize * sizeof(float);
    std::vector<float> result(matrixSize);

    float* deviceA = nullptr;
    float* deviceB = nullptr;
    float* deviceC = nullptr;

    CUDA_CHECK(cudaMalloc(&deviceA, bytes));
    CUDA_CHECK(cudaMalloc(&deviceB, bytes));
    CUDA_CHECK(cudaMalloc(&deviceC, bytes));

    CUDA_CHECK(cudaMemcpy(deviceA, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceB, b.data(), bytes, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((n + TILE_SIZE - 1) / TILE_SIZE, 
                       (n + TILE_SIZE - 1) / TILE_SIZE);

    blockGemmKernel<<<blocksPerGrid, threadsPerBlock>>>(deviceA, deviceB, deviceC, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(result.data(), deviceC, bytes, cudaMemcpyDeviceToHost));

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    return result;
}
