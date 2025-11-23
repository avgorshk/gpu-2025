#include "fft_cufft.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <vector>

__global__ void gpu_scale(cufftComplex* buf, int count, float scale) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < count) {
        buf[id].x *= scale;
        buf[id].y *= scale;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    int n = input.size() / (2 * batch);
    int total = n * batch;
    size_t bytes = total * sizeof(cufftComplex);

    cufftComplex* devicePtr = nullptr;
    cudaMalloc(&devicePtr, bytes);
    cudaMemcpy(devicePtr, input.data(), bytes, cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_C2C, batch);

    cufftExecC2C(plan, devicePtr, devicePtr, CUFFT_FORWARD);
    cufftExecC2C(plan, devicePtr, devicePtr, CUFFT_INVERSE);

    int tpb = 256;
    int blocks = (total + tpb - 1) / tpb;
    gpu_scale<<<blocks, tpb>>>(devicePtr, total, 1.0f / (float)n);

    std::vector<float> output(input.size());
    cudaMemcpy(output.data(), devicePtr, bytes, cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
    cudaFree(devicePtr);

    return output;
}
