#include "gelu_ocl.h"
#include <CL/cl.h>
#include <vector>
#include <string>
#include <iostream>

static const char* kernelSrc = R"CLC(
__kernel void gelu_ocl(__global const float* input,
                       __global float* output,
                       int n)
{
    int idx = get_global_id(0);
    if (idx >= n) return;

    float x = input[idx];
    float x3 = x * x * x;
    float c = 0.7978845608f;  // sqrt(2/pi)
    float alpha = 0.044715f;

    float v = x + alpha * x3;
    float z = c * v;

    // tanh approximation
    float e = exp(-2.0f * z);
    float t = (1.0f - e) / (1.0f + e);

    output[idx] = 0.5f * x * (1.0f + t);
}
)CLC";

std::vector<float> GeluOCL(const std::vector<float>& input, int platform)
{
    int n = (int)input.size();
    std::vector<float> output(n);

    if (n == 0)
        return output;

    cl_int err;

    // Platform
    cl_uint numPlatforms;
    clGetPlatformIDs(0, nullptr, &numPlatforms);

    std::vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);

    cl_platform_id plat = platforms[platform];

    // Device
    cl_device_id device;
    clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

    // Context
    cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);

    // Queue
    cl_command_queue queue = clCreateCommandQueue(ctx, device, 0, &err);

    // Program
    const char* src = kernelSrc;
    cl_program prog = clCreateProgramWithSource(ctx, 1, &src, nullptr, &err);
    clBuildProgram(prog, 1, &device, nullptr, nullptr, nullptr);

    // Kernel
    cl_kernel kernel = clCreateKernel(prog, "gelu_ocl", &err);

    // Buffers
    size_t bytes = sizeof(float) * n;

    cl_mem d_input  = clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytes, nullptr, &err);
    cl_mem d_output = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);

    // Copy input
    clEnqueueWriteBuffer(queue, d_input, CL_TRUE, 0, bytes, input.data(), 0, nullptr, nullptr);

    // Set args
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output);
    clSetKernelArg(kernel, 2, sizeof(int), &n);

    // Global size
    size_t global = n;

    // Run kernel
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
    clFinish(queue);

    // Copy back
    clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, bytes, output.data(), 0, nullptr, nullptr);

    // Cleanup
    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);
    clReleaseKernel(kernel);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    return output;
}
