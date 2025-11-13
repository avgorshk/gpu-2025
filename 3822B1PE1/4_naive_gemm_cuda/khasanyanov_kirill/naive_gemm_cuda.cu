#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>

__global__ void naiveGemmKernel(const float *a, const float *b, float *c,
                                int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n && col < n) {
    float sum = 0.0f;

    int k = 0;
    for (; k <= n - 4; k += 4) {

      float a0 = a[row * n + k];
      float a1 = a[row * n + k + 1];
      float a2 = a[row * n + k + 2];
      float a3 = a[row * n + k + 3];

      float b0 = b[k * n + col];
      float b1 = b[(k + 1) * n + col];
      float b2 = b[(k + 2) * n + col];
      float b3 = b[(k + 3) * n + col];

      sum += a0 * b0;
      sum += a1 * b1;
      sum += a2 * b2;
      sum += a3 * b3;
    }

    for (; k < n; ++k) {
      sum += a[row * n + k] * b[k * n + col];
    }

    c[row * n + col] = sum;
  }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float> &a,
                                 const std::vector<float> &b, int n) {
  if (n == 0) {
    return std::vector<float>();
  }

  std::vector<float> c(n * n);

  float *d_a = nullptr;
  float *d_b = nullptr;
  float *d_c = nullptr;

  size_t bytes = n * n * sizeof(float);

  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

  naiveGemmKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

  cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return c;
}
