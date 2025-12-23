#include "fft_cufft.h"
#include <cufft.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <memory>

// CUDA kernel for normalization
__global__ void normalize_kernel(cufftComplex *data, int n, int batch)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = n * batch;
    if (idx < total_elements)
    {
        data[idx].x /= n;
        data[idx].y /= n;
    }
}

std::vector<float> FffCUFFT(const std::vector<float> &input, int batch)
{
    if (input.empty() || batch <= 0)
    {
        throw std::invalid_argument("Invalid input or batch size");
    }

    size_t total_floats = input.size();
    if (total_floats % (2 * batch) != 0)
    {
        throw std::invalid_argument("Input size must be divisible by 2*batch");
    }
    int n = static_cast<int>(total_floats / (2 * batch));

    std::vector<float> output(total_floats);

    cufftComplex *d_data = nullptr;
    size_t data_size = n * batch * sizeof(cufftComplex);

    cudaMalloc(&d_data, data_size);

    cudaMemcpy(d_data, input.data(), data_size, cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_C2C, batch);

    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);

    cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);

    int total_elements = n * batch;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    normalize_kernel<<<grid_size, block_size>>>(d_data, n, batch);
    cudaDeviceSynchronize();

    cudaMemcpy(output.data(), d_data, data_size, cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
    cudaFree(d_data);

    return output;
}