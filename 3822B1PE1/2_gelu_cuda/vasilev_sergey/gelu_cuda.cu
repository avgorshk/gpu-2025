#include "gelu_cuda.h"

#include <cuda_runtime.h>
#include <vector>
#include <cstddef>

namespace
{

    constexpr float kSqrt2OverPiScaled = 1.595769122f;
    constexpr float kC = 0.044715f;

    __device__ __forceinline__ float gelu_element(float x)
    {
        float x3 = x * x * x;
        float z = kSqrt2OverPiScaled * (x + kC * x3);
        float s = 1.0f / (1.0f + __expf(-z));
        return x * s;
    }

    __global__ void gelu_kernel(const float *in, float *out, std::size_t n)
    {
        std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        std::size_t stride = blockDim.x * gridDim.x;

        for (std::size_t i = idx; i < n; i += stride)
        {
            out[i] = gelu_element(in[i]);
        }
    }

} // namespace

std::vector<float> GeluCUDA(const std::vector<float> &input)
{
    std::size_t n = input.size();
    std::vector<float> output(n);
    if (n == 0)
        return output;

    std::size_t bytes = n * sizeof(float);

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    cudaMemcpy(d_in, input.data(), bytes, cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    if (grid_size > 1024)
        grid_size = 1024;

    gelu_kernel<<<grid_size, block_size>>>(d_in, d_out, n);

    cudaMemcpy(output.data(), d_out, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);

    return output;
}
