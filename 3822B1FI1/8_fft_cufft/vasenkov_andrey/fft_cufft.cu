#include "fft_cufft.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <vector>

__global__ void ScaleComplexKernel(cufftComplex* data, int length, float factor) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= length) return;
    data[index].x *= factor;
    data[index].y *= factor;
}

std::vector<float> FffCUFFT(const std::vector<float>& inData, int batchSize) {
    int totalFloats = inData.size();
    int fftSize = totalFloats / (2 * batchSize);
    int complexSize = fftSize * batchSize;
    int bytes = totalFloats * sizeof(float);
    float scaleFactor = 1.0f / fftSize;
    std::vector<float> result(totalFloats);

    cufftComplex* devComplex;
    cudaMalloc(&devComplex, complexSize * sizeof(cufftComplex));
    cudaMemcpy(devComplex, inData.data(), bytes, cudaMemcpyHostToDevice);

    cufftHandle fftPlan;
    cufftPlan1d(&fftPlan, fftSize, CUFFT_C2C, batchSize);
    cufftExecC2C(fftPlan, devComplex, devComplex, CUFFT_FORWARD);
    cufftExecC2C(fftPlan, devComplex, devComplex, CUFFT_INVERSE);

    int threadsPerBlock = 256;
    int numBlocks = (complexSize + threadsPerBlock - 1) / threadsPerBlock;
    ScaleComplexKernel<<<numBlocks, threadsPerBlock>>>(devComplex, complexSize, scaleFactor);
    cudaDeviceSynchronize();

    cudaMemcpy(result.data(), devComplex, bytes, cudaMemcpyDeviceToHost);

    cufftDestroy(fftPlan);
    cudaFree(devComplex);

    return result;
}
