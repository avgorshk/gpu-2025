#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>

// Kernel CUDA naïf : chaque thread calcule un élément C[row, col]
__global__ void naive_gemm_kernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ c,
                                  int n)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x; // j
    int row = blockIdx.y * blockDim.y + threadIdx.y; // i

    if (row >= n || col >= n) return;

    float sum = 0.0f;
    int row_offset = row * n;

    // C[row, col] = sum_k A[row, k] * B[k, col]
    for (int k = 0; k < n; ++k) {
        float aik = a[row_offset + k];       // A[row, k]
        float bkj = b[k * n + col];         // B[k, col]
        sum += aik * bkj;
    }

    c[row_offset + col] = sum;
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n)
{
    const size_t size = static_cast<size_t>(n) * n;
    std::vector<float> c(size, 0.0f);

    if (n <= 0 || a.size() != size || b.size() != size) {
        // tailles invalides -> retourne un résultat zéro
        return c;
    }

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    const size_t bytes = size * sizeof(float);

    // Allocation mémoire sur le GPU
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Copie host -> device
    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_c, 0, bytes);

    // Configuration du lancement (grid 2D, block 2D)
    const int BLOCK_SIZE = 16;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Lancement du kernel
    naive_gemm_kernel<<<grid, block>>>(d_a, d_b, d_c, n);

    // Attendre la fin des calculs
    cudaDeviceSynchronize();

    // Copie device -> host
    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    // Libération mémoire GPU
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}
