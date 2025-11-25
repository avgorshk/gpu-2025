#include "fft_cufft.h"
#include <cuda_runtime.h>
#include <cufft.h>

__global__ void scaleSignal(float* __restrict__ data, int len, float factor) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int quad_idx = tid >> 2;
    
    if (quad_idx < (len >> 2)) {
        float4* vec_ptr = reinterpret_cast<float4*>(data);
        float4 vals = __ldg(&vec_ptr[quad_idx]);
        vals.x *= factor;
        vals.y *= factor;
        vals.z *= factor;
        vals.w *= factor;
        vec_ptr[quad_idx] = vals;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) 
{
    cudaDeviceProp gpu_info;
    cudaGetDeviceProperties(&gpu_info, 0);

    const int total_floats = input.size();
    std::vector<float> processed(total_floats);

    const int complex_per_batch = total_floats / batch / 2;
    const int mem_bytes = sizeof(cufftComplex) * complex_per_batch * batch;
    const int block_threads = gpu_info.maxThreadsPerBlock;
    const int grid_blocks = (total_floats + block_threads - 1) / block_threads;

    cufftComplex* gpu_data;
    cudaMalloc(&gpu_data, mem_bytes);
    cudaMemcpy(gpu_data, input.data(), mem_bytes, cudaMemcpyHostToDevice);

    cufftHandle fft_plan;
    cufftPlan1d(&fft_plan, complex_per_batch, CUFFT_C2C, batch);

    cufftExecC2C(fft_plan, gpu_data, gpu_data, CUFFT_FORWARD);
    cufftExecC2C(fft_plan, gpu_data, gpu_data, CUFFT_INVERSE);

    scaleSignal<<<grid_blocks, block_threads>>>(
        reinterpret_cast<float*>(gpu_data), 
        total_floats, 
        1.0f / complex_per_batch
    );

    cudaMemcpy(processed.data(), gpu_data, mem_bytes, cudaMemcpyDeviceToHost);
    
    cufftDestroy(fft_plan);
    cudaFree(gpu_data);

    return processed;
}