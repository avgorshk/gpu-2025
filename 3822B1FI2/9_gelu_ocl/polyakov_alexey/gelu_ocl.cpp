#include "gelu_ocl.h"
#include <CL/opencl.h>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <cmath>

const char* gelu_kernel_source = R"(
__kernel void gelu(__global const float* input, __global float* output, const int n) {
    int idx = get_global_id(0);
    if (idx >= n) return;
    
    float x = input[idx];
    float cdf = 0.5f * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
    output[idx] = x * cdf;
}
)";

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    cl_uint num_platforms;
    clGetPlatformIDs(0, NULL, &num_platforms);
    if (platform >= static_cast<int>(num_platforms)) {
        throw std::runtime_error("Platform index out of bounds");
    }

    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), NULL);

    cl_device_id device_id;
    clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

    cl_int err;
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create context");

    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create command queue");

    cl_program program = clCreateProgramWithSource(context, 1, &gelu_kernel_source, NULL, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create program");

    err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        char build_log[4096];
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(build_log), build_log, NULL);
        throw std::runtime_error(std::string("Build failed: ") + build_log);
    }

    cl_kernel kernel = clCreateKernel(program, "gelu", &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create kernel");

    size_t n = input.size();
    size_t buffer_size = n * sizeof(float);

    cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buffer_size, (void*)input.data(), &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create input buffer");

    cl_mem output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, buffer_size, NULL, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create output buffer");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buffer);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &n);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to set kernel arguments");

    size_t global_size = (n + 255) / 256 * 256;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to enqueue kernel");

    std::vector<float> result(n);
    err = clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, buffer_size, result.data(), 0, NULL, NULL);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to read output buffer");

    clReleaseMemObject(input_buffer);
    clReleaseMemObject(output_buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return result;
}