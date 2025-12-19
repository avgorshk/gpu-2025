#include "gelu_ocl.h"

#include <vector>
#include <cmath>
#include <CL/cl.h>

const char* kernel_str = R"(
    __kernel void gelu_kernel(__global const float* input, __global float* output, int n) {
        int idx = get_global_id(0);
        if (idx >= n) return;
        float x = input[idx];
        float x3 = x*x*x;
        const float sqrt2pi = sqrt(2.0f / M_PI);
        output[idx] = 0.5f * x * (1.0f + tanh(sqrt2pi * (x + 0.044715f * x3)));
    })";

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    size_t n = input.size();

    cl_platform_id platform_id;
    clGetPlatformIDs(1, &platform_id, nullptr);

    cl_device_id device;
    clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, nullptr, nullptr);
    
    cl_program program = clCreateProgramWithSource(context, 1, &kernel_str, nullptr, nullptr);
    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    cl_kernel kernel = clCreateKernel(program, "gelu_kernel", nullptr);

    cl_mem d_input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * sizeof(float), (void*)input.data(), nullptr);
    cl_mem d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n * sizeof(float), nullptr, nullptr);

    std::vector<float> output(n, 0.0f);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output);
    clSetKernelArg(kernel, 2, sizeof(int), &n);

    size_t global_size = n;
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr);
    clFinish(queue);

    clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, n * sizeof(float), output.data(), 0, nullptr, nullptr);

    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return output;
}