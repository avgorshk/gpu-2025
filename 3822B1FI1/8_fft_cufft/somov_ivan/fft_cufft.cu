#include "fft_cufft.h"

#include <cuda_runtime.h>
#include <cufft.h>

__global__ void normalize(cufftComplex* input, int complexes, float n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < complexes) {
        cufftComplex complex = input[i];
        input[i].x = complex.x * n;
        input[i].y = complex.y * n;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    std::vector<float> result(input.size());

    const int floats = input.size();
    const int complexes = input.size() / 2;

    cufftComplex* data;
    cudaMalloc((void**)&data, complexes * sizeof(cufftComplex));
    cudaMemcpy(data, input.data(), floats * sizeof(float), cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlan1d(&plan, complexes / batch, CUFFT_C2C, batch);
    cufftExecC2C(plan, data, data, CUFFT_FORWARD);

    int block_size;
    int num_blocks;
    cudaOccupancyMaxPotentialBlockSize(&num_blocks, &block_size, normalize, 0, 0);
    num_blocks = (floats + block_size - 1) / block_size;
    normalize << < num_blocks, block_size >> > (data, complexes, 1.0f / (complexes / batch));

    cufftExecC2C(plan, data, data, CUFFT_INVERSE);
    cufftDestroy(plan);

    cudaMemcpy(result.data(), data, floats * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(data);

    return result;
}