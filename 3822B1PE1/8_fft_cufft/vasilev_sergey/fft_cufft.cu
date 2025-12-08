#include "fft_cufft.h"

#include <cuda_runtime.h>
#include <cufft.h>
#include <cstddef>
#include <vector>

__global__ void normalize_kernel(cufftComplex *data, int total, float scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total)
    {
        data[idx].x *= scale;
        data[idx].y *= scale;
    }
}

std::vector<float> FffCUFFT(const std::vector<float> &input, int batch)
{
    std::size_t total_floats = input.size();
    if (batch <= 0 || total_floats == 0)
    {
        return {};
    }

    std::size_t per_batch = total_floats / static_cast<std::size_t>(batch);
    if (per_batch % 2 != 0)
    {
        return {};
    }

    int n = static_cast<int>(per_batch / 2);
    std::size_t total_complex = static_cast<std::size_t>(n) * batch;

    std::vector<float> output(total_floats);

    cufftComplex *d_data;
    cudaMalloc(&d_data, total_complex * sizeof(cufftComplex));

    cudaMemcpy(d_data,
               input.data(),
               total_floats * sizeof(float),
               cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_C2C, batch);

    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);

    float scale = 1.0f / static_cast<float>(n);
    int threads = 256;
    int blocks = static_cast<int>((total_complex + threads - 1) / threads);
    normalize_kernel<<<blocks, threads>>>(d_data,
                                          static_cast<int>(total_complex),
                                          scale);

    cufftDestroy(plan);

    cudaMemcpy(output.data(),
               d_data,
               total_floats * sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaFree(d_data);

    return output;
}
