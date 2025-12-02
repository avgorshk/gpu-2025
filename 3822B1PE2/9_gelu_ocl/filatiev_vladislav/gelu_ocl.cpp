#include "gelu_ocl.h"
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <string>

const char* gelu_kernel_code = R"(
__kernel void gelu_kernel(__global float* input, __global float* output, int n) {
    int idx = get_global_id(0);
    if (idx < n) {
        float x = input[idx];
        float x3 = x * x * x;
        float inner = 0.7978845608028654f * (x + 0.044715f * x3);
        float exp_val = exp(2.0f * inner);
        float tanh_approx = (exp_val - 1.0f) / (exp_val + 1.0f);
        output[idx] = 0.5f * x * (1.0f + tanh_approx);
    }
}
)";

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    if (input.empty()) return std::vector<float>();

    cl_int err;
    cl_uint num_platforms;

    clGetPlatformIDs(0, NULL, &num_platforms);
    if (num_platforms == 0) throw std::runtime_error("no opencl platforms");

    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), NULL);

    if (platform < 0 || platform >= (int)num_platforms) {
        throw std::runtime_error("invalid platform");
    }

    cl_device_id device;
    err = clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) throw std::runtime_error("no gpu device");

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("context failed");

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS) {
        clReleaseContext(context);
        throw std::runtime_error("queue failed");
    }

    cl_program program = clCreateProgramWithSource(context, 1, &gelu_kernel_code, NULL, &err);
    if (err != CL_SUCCESS) {
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error("program failed");
    }

    err = clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), NULL);
        std::string error = "build failed: " + std::string(log.data());

        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error(error);
    }

    cl_kernel kernel = clCreateKernel(program, "gelu_kernel", &err);
    if (err != CL_SUCCESS) {
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error("kernel failed");
    }

    size_t n = input.size();
    cl_mem d_input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        n * sizeof(float), (void*)input.data(), &err);
    if (err != CL_SUCCESS) {
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error("input buffer failed");
    }

    cl_mem d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(d_input);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error("output buffer failed");
    }

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &n);

    if (err != CL_SUCCESS) {
        clReleaseMemObject(d_output);
        clReleaseMemObject(d_input);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error("kernel args failed");
    }

    size_t local = 256;
    size_t global = ((n + local - 1) / local) * local;

    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(d_output);
        clReleaseMemObject(d_input);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error("kernel exec failed");
    }

    std::vector<float> result(n);
    err = clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, n * sizeof(float), result.data(), 0, NULL, NULL);

    clReleaseMemObject(d_output);
    clReleaseMemObject(d_input);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return result;
}