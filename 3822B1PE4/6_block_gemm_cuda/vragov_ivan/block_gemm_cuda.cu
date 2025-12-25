#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_SIZE 16

// CUDA Kernel for Block GEMM using Shared Memory
__global__ void BlockGemmKernel(const float *__restrict__ A,
                                const float *__restrict__ B,
                                float *__restrict__ C, int n) {
  // blockRow and blockCol for this sub-block
  int blockRow = blockIdx.y;
  int blockCol = blockIdx.x;

  // Thread index within the sub-block
  int row = threadIdx.y;
  int col = threadIdx.x;

  // Calculate accumulation in registers
  float Cvalue = 0.0f;

  // Loop over all sub-matrices of A and B required to compute the block Csub
  // A moves horizontally, B moves vertically
  for (int m = 0; m < n / BLOCK_SIZE; ++m) {

    // Shared memory for current sub-matrices
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Load As from global memory
    // A sub-matrix: row `blockRow` of blocks, `m`-th block
    // Global Row: blockRow*BLOCK_SIZE + row
    // Global Col: m*BLOCK_SIZE + col
    As[row][col] =
        A[(blockRow * BLOCK_SIZE + row) * n + (m * BLOCK_SIZE + col)];

    // Load Bs from global memory
    // B sub-matrix: `m`-th block of rows, `blockCol` of blocks
    // Global Row: m*BLOCK_SIZE + row
    // Global Col: blockCol*BLOCK_SIZE + col
    Bs[row][col] =
        B[(m * BLOCK_SIZE + row) * n + (blockCol * BLOCK_SIZE + col)];

    // Synchronize to ensure the sub-matrices are loaded
    __syncthreads();

// Multiply As and Bs
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      Cvalue += As[row][k] * Bs[k][col];
    }

    // Synchronize to ensure computations are done before loading next
    // sub-matrices
    __syncthreads();
  }

  // Write back result
  // Global Row: blockRow*BLOCK_SIZE + row
  // Global Col: blockCol*BLOCK_SIZE + col
  C[(blockRow * BLOCK_SIZE + row) * n + (blockCol * BLOCK_SIZE + col)] = Cvalue;
}

std::vector<float> BlockGemmCUDA(const std::vector<float> &a,
                                 const std::vector<float> &b, int n) {
  // 1. Allocate host memory for result
  std::vector<float> c(n * n);
  size_t sizeInBytes = n * n * sizeof(float);

  // 2. Allocate device memory
  float *d_a, *d_b, *d_c;
  cudaError_t err;

  err = cudaMalloc((void **)&d_a, sizeInBytes);
  if (err != cudaSuccess)
    std::cerr << "CUDA Error Alloc A: " << cudaGetErrorString(err) << std::endl;

  err = cudaMalloc((void **)&d_b, sizeInBytes);
  if (err != cudaSuccess)
    std::cerr << "CUDA Error Alloc B: " << cudaGetErrorString(err) << std::endl;

  err = cudaMalloc((void **)&d_c, sizeInBytes);
  if (err != cudaSuccess)
    std::cerr << "CUDA Error Alloc C: " << cudaGetErrorString(err) << std::endl;

  // 3. Copy data to device
  cudaMemcpy(d_a, a.data(), sizeInBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b.data(), sizeInBytes, cudaMemcpyHostToDevice);

  // 4. Launch Kernel
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid(n / BLOCK_SIZE, n / BLOCK_SIZE);

  BlockGemmKernel<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, n);

  // Check for kernel launch errors
  err = cudaGetLastError();
  if (err != cudaSuccess)
    std::cerr << "CUDA Error Kernel: " << cudaGetErrorString(err) << std::endl;

  // 5. Copy result back
  cudaMemcpy(c.data(), d_c, sizeInBytes, cudaMemcpyDeviceToHost);

  // 6. Free device memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return c;
}
