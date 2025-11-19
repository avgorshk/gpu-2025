#include "gelu_ocl.h"
#include <CL/cl.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <string>

const std::string GELU_KERNEL_SOURCE = R"(
__kernel void gelu_kernel(__global const float* input, __global float* output, int n) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coef = 0.044715f;
    
    int idx = get_global_id(0);
    if (idx < n) {
        float x = input[idx];
        float x3 = x * x * x;
        float inner = sqrt_2_over_pi * (x + coef * x3);
        output[idx] = 0.5f * x * (1.0f + tanh(inner));
    }
}
)";

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    if (input.empty()) {
        return std::vector<float>();
    }

    cl_int err;
    cl_uint num_platforms;
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        throw std::runtime_error("No OpenCL platforms found");
    }

    std::vector<cl_platform_id> platforms(num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms.data(), NULL);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to get platforms");
    }

    if (platform >= static_cast<int>(num_platforms)) {
        throw std::runtime_error("Platform index out of range");
    }
    cl_platform_id platform_id = platforms[platform];
    cl_device_id device_id;
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("No GPU device found");
    }
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create context");
    }
    cl_command_queue queue = clCreateCommandQueue(context, device_id, 0, &err);
    if (err != CL_SUCCESS) {
        clReleaseContext(context);
        throw std::runtime_error("Failed to create command queue");
    }

    const char* source_str = GELU_KERNEL_SOURCE.c_str();
    size_t source_size = GELU_KERNEL_SOURCE.size();
    cl_program program = clCreateProgramWithSource(context, 1, &source_str, &source_size, &err);
    if (err != CL_SUCCESS) {
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error("Failed to create program");
    }
    err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log.data(), NULL);
        std::string error_msg = "Build failed: " + std::string(log.data());

        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error(error_msg);
    }

    cl_kernel kernel = clCreateKernel(program, "gelu_kernel", &err);
    if (err != CL_SUCCESS) {
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error("Failed to create kernel");
    }

    size_t n = input.size();
    cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        n * sizeof(float), (void*)input.data(), &err);
    if (err != CL_SUCCESS) {
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error("Failed to create input buffer");
    }

    cl_mem output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        n * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(input_buffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error("Failed to create output buffer");
    }
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buffer);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &n);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(output_buffer);
        clReleaseMemObject(input_buffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error("Failed to set kernel arguments");
    }

    size_t global_size = n;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(output_buffer);
        clReleaseMemObject(input_buffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error("Failed to execute kernel");
    }

    std::vector<float> output(n);
    err = clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0,
        n * sizeof(float), output.data(), 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(output_buffer);
        clReleaseMemObject(input_buffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error("Failed to read result");
    }

    clReleaseMemObject(output_buffer);
    clReleaseMemObject(input_buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return output;
}