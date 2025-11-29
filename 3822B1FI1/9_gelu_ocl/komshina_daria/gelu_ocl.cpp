#include "gelu_ocl.h"
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <cmath>

const char* geluKernelSource = R"CLC(
__kernel void gelu_kernel(__global const float* input, __global float* output, const int n) {
    int i = get_global_id(0);
    if (i < n) {
        float x = input[i];
        float x3 = x * x * x;
        float tanh_arg = 0.79788456f * (x + 0.044715f * x3); // sqrt(2/pi)
        float tanh_res = tanh(tanh_arg);
        output[i] = 0.5f * x * (1.0f + tanh_res);
    }
}
)CLC";

std::vector<float> GeluOCL(const std::vector<float>& input, int platform_index) {
    std::vector<float> output(input.size(), 0.0f);
    cl_int err;

    if (input.empty()) return output;

    cl_uint numPlatforms;
    err = clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (err != CL_SUCCESS || platform_index >= numPlatforms) {
        std::cerr << "Failed to get platforms or invalid platform index\n";
        return output;
    }
    std::vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    cl_platform_id platform = platforms[platform_index];

    cl_uint numDevices;
    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &numDevices);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to get GPU device\n";
        return output;
    }

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);

    size_t bytes = input.size() * sizeof(float);
    cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        bytes, (void*)input.data(), &err);
    cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);

    cl_program program = clCreateProgramWithSource(context, 1, &geluKernelSource, nullptr, &err);
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::cerr << "Build failed:\n" << log.data() << "\n";
        return output;
    }

    cl_kernel kernel = clCreateKernel(program, "gelu_kernel", &err);
    int n = static_cast<int>(input.size());
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
    clSetKernelArg(kernel, 2, sizeof(int), &n);

    size_t globalSize = ((input.size() + 255) / 256) * 256;
    size_t localSize = 256;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, &localSize,
        0, nullptr, nullptr);

    clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, bytes, output.data(), 0, nullptr, nullptr);

    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return output;
}
