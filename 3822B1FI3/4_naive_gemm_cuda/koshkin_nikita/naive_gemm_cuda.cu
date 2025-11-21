#include "naive_gemm_cuda.h"

#define TILE 16
#define BLOCK_ROWS 4


template<typename idx_t>
__global__
void matmul_tiled_idx(const float* __restrict__ A,
                      const float* __restrict__ B,
                      float* __restrict__ C,
                      idx_t N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    unsigned bx = blockIdx.x, by = blockIdx.y;
    unsigned tx = threadIdx.x, ty = threadIdx.y;

    idx_t row_base = (idx_t)by * (idx_t)TILE + (idx_t)ty;
    idx_t col = (idx_t)bx * (idx_t)TILE + (idx_t)tx;

    float acc[BLOCK_ROWS];
    #pragma unroll
    for (int i = 0; i < BLOCK_ROWS; ++i) {
        acc[i] = 0.0f;
    }

    for (idx_t m = 0; m < N; m += TILE) {
        #pragma unroll
        for (int i = 0; i < BLOCK_ROWS; ++i) {
            idx_t r = row_base + (idx_t)(i * BLOCK_ROWS);
            idx_t k = m + (idx_t)tx;
            if (r < N && k < N) {
                As[ty + i * BLOCK_ROWS][tx] = A[(size_t)r * (size_t)N + (size_t)k];
            } else {
                As[ty + i * BLOCK_ROWS][tx] = 0.0f;
            }
        }

        #pragma unroll
        for (int i = 0; i < BLOCK_ROWS; ++i) {
            idx_t kb = m + (idx_t)(ty + i * BLOCK_ROWS);
            if (kb < N && col < N) {
                Bs[ty + i * BLOCK_ROWS][tx] = B[(size_t)kb * (size_t)N + (size_t)col];
            } else {
                Bs[ty + i * BLOCK_ROWS][tx] = 0.0f;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            float b = Bs[k][tx];
            #pragma unroll
            for (int i = 0; i < BLOCK_ROWS; ++i) {
                float a = As[ty + i * BLOCK_ROWS][k];
                acc[i] += a * b;
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < BLOCK_ROWS; ++i) {
        idx_t r = row_base + (idx_t)(i * BLOCK_ROWS);
        if (r < N && col < N) {
            C[(size_t)r * (size_t)N + (size_t)col] = acc[i];
        }
    }
}

void launch_matmul(const float* d_A, const float* d_B, float* d_C, size_t N,
                   cudaStream_t stream = 0) {
    dim3 block(TILE, BLOCK_ROWS);
    dim3 grid( (N + TILE - 1) / TILE, (N + TILE - 1) / TILE );

    const size_t signed32_threshold = 46340;
    const size_t uint32_threshold   = 65535;

    if (N <= signed32_threshold) {
        matmul_tiled_idx<uint32_t><<<grid, block, 0, stream>>>(d_A, d_B, d_C, static_cast<uint32_t>(N));
    } else {
        matmul_tiled_idx<size_t><<<grid, block, 0, stream>>>(d_A, d_B, d_C, static_cast<size_t>(N));
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    size_t N = static_cast<size_t>(n);
    size_t elems = N * N;

    size_t bytes = elems * sizeof(float);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

    std::vector<float> C(elems);

    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);

    cudaMemcpy(d_A, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b.data(), bytes, cudaMemcpyHostToDevice);

    launch_matmul(d_A, d_B, d_C, N, 0);

    cudaDeviceSynchronize();

    cudaMemcpy(C.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return C;
}