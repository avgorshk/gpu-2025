#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <vector>

#define ROOT_TWO_OVER_PI 0.7978845608028654f
#define CUBIC_COEFFICIENT 0.044715f
#define ONE_HALF 0.5f

__device__ float calculateHyperbolicTangent(float v) {
    float exponent = __expf(2.0f * v);
    return (exponent - 1.0f) / (exponent + 1.0f);
}

__global__ void executeGeluCalculation(const float* source_data,
                                       float* destination_data,
                                       int element_count) {
    int linear_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (linear_index >= element_count) return;

    float x = source_data[linear_index];
    float x3 = x * x * x;

    float t = ROOT_TWO_OVER_PI * (x + CUBIC_COEFFICIENT * x3);
    float th = calculateHyperbolicTangent(t);

    destination_data[linear_index] = ONE_HALF * x * (1.0f + th);
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int total_elements = static_cast<int>(input.size());
    std::vector<float> destination(total_elements);
    if (total_elements == 0) return destination;

    static float* device_source_buffer = nullptr;
    static float* device_destination_buffer = nullptr;
    static int current_capacity = 0;

    cudaError_t error_status;

    if (device_source_buffer == nullptr || current_capacity < total_elements) {
        if (device_source_buffer) cudaFree(device_source_buffer);
        if (device_destination_buffer) cudaFree(device_destination_buffer);

        error_status = cudaMalloc(&device_source_buffer,
                                  total_elements * sizeof(float));
        if (error_status != cudaSuccess) return destination;

        error_status = cudaMalloc(&device_destination_buffer,
                                  total_elements * sizeof(float));
        if (error_status != cudaSuccess) {
            cudaFree(device_source_buffer);
            device_source_buffer = nullptr;
            return destination;
        }

        current_capacity = total_elements;
    }

    cudaStream_t async_stream;
    cudaStreamCreate(&async_stream);

    cudaMemcpyAsync(device_source_buffer, input.data(),
                    total_elements * sizeof(float),
                    cudaMemcpyHostToDevice, async_stream);

    int thread_block_size = 256;
    int grid_block_count =
        (total_elements + thread_block_size - 1) / thread_block_size;

    executeGeluCalculation<<<grid_block_count, thread_block_size, 0,
                             async_stream>>>(
        device_source_buffer, device_destination_buffer, total_elements);

    cudaMemcpyAsync(destination.data(), device_destination_buffer,
                    total_elements * sizeof(float),
                    cudaMemcpyDeviceToHost, async_stream);

    cudaStreamSynchronize(async_stream);
    cudaStreamDestroy(async_stream);

    return destination;
}
