#include "cuda_runtime.h"
#include "gelu_cuda.h"
#include <vector>
#include <chrono>

#define a 0.7978845608028654f
#define b 0.044715f
#define ui unsigned int

__global__ void gelu_kernel(float* c, const float* input, const size_t size)
{
    ui i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        const float x = input[i];
        const float x_cube = x * x * x;
        const float tanh_args = a * (x + b * x_cube);
        const float tanh_result = tanh(tanh_args);
        c[i] = 0.5f * x * (1.0f + tanh_result);
    }
}
std::vector<float> GeluCUDA(const std::vector<float>& input){
    const size_t size = input.size();
    std::vector<float> res(size);
    const float* input_ptr = input.data();
    float* input_cuda, * res_cuda;

    cudaMalloc(&input_cuda, size * sizeof(float));
    cudaMalloc(&res_cuda, size * sizeof(float));
    cudaMemcpy(input_cuda, input_ptr, size * sizeof(float), cudaMemcpyHostToDevice);

    constexpr int THREADS_PER_BLOCK = 256;
    int BLOCKS = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    gelu_kernel <<<BLOCKS, THREADS_PER_BLOCK >>> (res_cuda, input_cuda, size);

    cudaDeviceSynchronize();

    cudaMemcpy(res.data(), res_cuda, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(input_cuda);
    cudaFree(res_cuda);
    return res;
}