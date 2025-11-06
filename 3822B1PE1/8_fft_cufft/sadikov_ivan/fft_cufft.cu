#include "fft_cufft.h"

#include <cuda_runtime.h>
#include <cufft.h>

__global__ void normalize(cufftComplex* data, int size, float scale)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    data[idx].x *= scale;
    data[idx].y *= scale;
  }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) 
{
    std::vector<float> result(input.size());
    int size = static_cast<int>(input.size() / (2 * batch));
    constexpr int threadsCount {256};
    const int blocksCount = (size + threadsCount - 1) / threadsCount;

    cufftComplex* buffer;
    cudaMalloc(&buffer, size * sizeof(cufftComplex) * batch);
    cudaMemcpy(buffer, input.data(), size * sizeof(cufftComplex) * batch, cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlan1d(&plan, size, CUFFT_C2C, batch);
    cufftExecC2C(plan, buffer, buffer, CUFFT_FORWARD);
    cufftExecC2C(plan, buffer, buffer, CUFFT_INVERSE);

    normalize<<<blocksCount, threadsCount>>>(buffer, size * batch, 1.0f/size);

    cudaMemcpy(result.data(), buffer, size * sizeof(cufftComplex) * batch, cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
    cudaFree(buffer);

    return result;
}