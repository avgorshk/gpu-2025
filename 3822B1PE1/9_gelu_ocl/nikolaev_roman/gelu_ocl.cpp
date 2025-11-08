#include "gelu_ocl.h"
#include <CL/cl.h>
#include <vector>
#include <stdexcept>
#include <string>

const std::string GELU_KERNEL = R"(
__kernel void gelu(__global const float* input, __global float* output, int n) {
    int idx = get_global_id(0);
    if (idx >= n) return;
    float x = input[idx];
    float sqrt_2_over_pi = 0.7978845608028654f;
    float coeff = 0.044715f;
    float x_cubed = x * x * x;
    float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
    float exp_2inner = exp(2.0f * inner);
    float tanh_inner = (exp_2inner - 1.0f) / (exp_2inner + 1.0f);
    output[idx] = 0.5f * x * (1.0f + tanh_inner);
}
)";

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    cl_int err;
    cl_uint num_platforms;
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) throw std::runtime_error("No OpenCL platforms found");
    std::vector<cl_platform_id> platforms(num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms.data(), NULL);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to get platform IDs");
    if (platform >= (int)num_platforms) platform = 0;

    cl_uint num_devices;
    err = clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
    if (err != CL_SUCCESS || num_devices == 0) throw std::runtime_error("No GPU devices found");
    cl_device_id device;
    err = clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to get device ID");

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create context");
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS) {
        clReleaseContext(context);
        throw std::runtime_error("Failed to create command queue");
    }

    const char* kernel_source = GELU_KERNEL.c_str();
    cl_program program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &err);
    if (err != CL_SUCCESS) {
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error("Failed to create program");
    }

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), NULL);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error("Failed to build program: " + std::string(log.data()));
    }

    cl_kernel kernel = clCreateKernel(program, "gelu", &err);
    if (err != CL_SUCCESS) {
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error("Failed to create kernel");
    }

    size_t n = input.size();
    cl_mem d_input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * sizeof(float), (void*)input.data(), &err);
    if (err != CL_SUCCESS) {
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error("Failed to create input buffer");
    }
    cl_mem d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(d_input);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error("Failed to create output buffer");
    }

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &n);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(d_input);
        clReleaseMemObject(d_output);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error("Failed to set kernel arguments");
    }

    size_t local_size = 256;
    size_t global_size = ((n + local_size - 1) / local_size) * local_size;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(d_input);
        clReleaseMemObject(d_output);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error("Failed to enqueue kernel");
    }

    std::vector<float> result(n);
    err = clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, n * sizeof(float), result.data(), 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(d_input);
        clReleaseMemObject(d_output);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error("Failed to read output buffer");
    }

    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return result;
}
