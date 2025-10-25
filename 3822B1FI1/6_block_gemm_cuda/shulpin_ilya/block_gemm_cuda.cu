#include "block_gemm_cuda.h"

template <int TILE, int PAD = 1, typename idx_t = size_t>
__global__ void block_gemm_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  idx_t n) {
    __shared__ float As[TILE][TILE + PAD];
    __shared__ float Bs[TILE][TILE + PAD];

    unsigned bx = blockIdx.x, by = blockIdx.y;
    unsigned tx = threadIdx.x, ty = threadIdx.y;

    idx_t row = (idx_t)by * (idx_t)TILE + (idx_t)ty;
    idx_t col = (idx_t)bx * (idx_t)TILE + (idx_t)tx;

    float sum = 0.0f;

    for (idx_t t = 0; t < n; t += (idx_t)TILE) {
        idx_t a_col = t + (idx_t)tx;
        if (row < n && a_col < n) {
            As[ty][tx] = A[(size_t)row * (size_t)n + (size_t)a_col];
        } else {
            As[ty][tx] = 0.0f;
        }

        idx_t b_row = t + (idx_t)ty;
        if (b_row < n && col < n) {
            Bs[ty][tx] = B[(size_t)b_row * (size_t)n + (size_t)col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        C[(size_t)row * (size_t)n + (size_t)col] = sum;
    }
}

template <int TILE = 16, int PAD = 1, typename idx_t = size_t>
std::vector<float> BlockGemmCUDA_T(const std::vector<float>& a,
                                   const std::vector<float>& b,
                                   idx_t n) {
    size_t N = static_cast<size_t>(n);
    size_t elems = N * N;
    size_t bytes = elems * sizeof(float);

    std::vector<float> C(elems, 0.0f);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);

    cudaMemcpy(d_A, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block((unsigned)TILE, (unsigned)TILE);
    dim3 grid((unsigned)((N + TILE - 1) / TILE), (unsigned)((N + TILE - 1) / TILE));


    block_gemm_kernel<TILE, PAD, idx_t><<<grid, block>>>(d_A, d_B, d_C, n);
    cudaGetLastError();
    cudaDeviceSynchronize();

    cudaMemcpy(C.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return C;
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    const size_t uint32_threshold = 0xFFFFFFFFu;
    size_t N = static_cast<size_t>(n);

    if (N <= 0xFFFFFFFFu) {
        return BlockGemmCUDA_T<16, 1, uint32_t>(a, b, static_cast<uint32_t>(n));
    } else {
        return BlockGemmCUDA_T<16, 1, size_t>(a, b, static_cast<size_t>(n));
    }
}