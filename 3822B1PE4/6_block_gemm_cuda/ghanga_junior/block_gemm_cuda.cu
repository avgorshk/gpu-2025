#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>

#define BLOCK_SIZE 16

// Kernel GEMM avec tiling et mémoire partagée
__global__ void block_gemm_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int n)
{
    __shared__ float Asub[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bsub[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Nombre de tiles dans la dimension K
    int tiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int t = 0; t < tiles; ++t) {

        int A_col = t * BLOCK_SIZE + threadIdx.x;
        int B_row = t * BLOCK_SIZE + threadIdx.y;

        // Charger le tile de A → Asub
        if (row < n && A_col < n)
            Asub[threadIdx.y][threadIdx.x] = A[row * n + A_col];
        else
            Asub[threadIdx.y][threadIdx.x] = 0.0f;

        // Charger le tile de B → Bsub
        if (B_row < n && col < n)
            Bsub[threadIdx.y][threadIdx.x] = B[B_row * n + col];
        else
            Bsub[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();  // Sync threads in block

        // Accumuler les produits du tile
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += Asub[threadIdx.y][k] * Bsub[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Écrire le résultat
    if (row < n && col < n)
        C[row * n + col] = sum;
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n)
{
    const size_t size = static_cast<size_t>(n) * n;
    std::vector<float> c(size, 0.0f);

    if (n <= 0 || a.size() != size || b.size() != size)
        return c;

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    size_t bytes = size * sizeof(float);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_c, 0, bytes);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    block_gemm_kernel<<<grid, block>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}
