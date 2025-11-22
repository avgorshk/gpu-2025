#include "block_gemm_cuda.h"

const int BLOCK_SIZE = 16;

__global__ void block_gemm_kernel(const float* __restrict__ a,
	                              const float* __restrict__ b,
	                              float* __restrict__ c,
	                              int n) {

	__shared__ float a_shared[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float b_shared[BLOCK_SIZE][BLOCK_SIZE];

	int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
	int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

	float sum = 0.0f;
	int num_blocks = n / BLOCK_SIZE;

	for (int block_k = 0; block_k < num_blocks; ++block_k) {
		a_shared[threadIdx.y][threadIdx.x] = a[row * n + block_k * BLOCK_SIZE + threadIdx.x];
		b_shared[threadIdx.y][threadIdx.x] = b[(block_k * BLOCK_SIZE + threadIdx.y) * n + col];

		__syncthreads();

#pragma unroll
		for (int k = 0; k < BLOCK_SIZE; ++k) {
			sum += a_shared[threadIdx.y][k] * b_shared[k][threadIdx.x];
		}

		__syncthreads();
	}

	c[row * n + col] = sum;
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
                                    
    std::vector<float> c(n * n);
    
    float *d_a, *d_b, *d_c;
    size_t size = n * n * sizeof(float);
    
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice);
    
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((n + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                 (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    block_gemm_kernel<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
    
    cudaMemcpy(c.data(), d_c, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return c;
}