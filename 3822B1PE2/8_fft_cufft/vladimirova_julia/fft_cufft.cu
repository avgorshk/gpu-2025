#include "fft_cufft.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <vector>

__global__ void Norm3Ts(cufftComplex* input, int n, float f) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;
    input[idx].x *= f;
    input[idx].y *= f;
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    int i_size = input.size();
    int n = i_size / (2 * batch);
    int comp_size = n * batch;
    int d_size = i_size * sizeof(float);
    float f = 1.0 / n;
    std::vector<float> output(i_size);


    cufftComplex* d_comp_input;
    cudaMalloc(&d_comp_input, comp_size * sizeof(cufftComplex));
    cudaMemcpy(d_comp_input, input.data(), d_size, cudaMemcpyHostToDevice);


    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_C2C, batch);
    //1
    cufftExecC2C(plan, d_comp_input, d_comp_input, CUFFT_FORWARD);
    //2
    cufftExecC2C(plan, d_comp_input, d_comp_input, CUFFT_INVERSE);

    //3
    int threads = 256;
    int block = (comp_size + threads - 1) / threads;
    Norm3Ts<<<block, threads>>>(d_comp_input, comp_size, f);
    cudaDeviceSynchronize();

    cudaMemcpy(output.data(), d_comp_input, d_size, cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
    cudaFree(d_comp_input);

    return output;
}