#include "gelu_ocl.h"
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <string>
#include <cmath>

static const char* gelu_kernel_src = R"CLC(
__kernel void gelu_kernel(__global const float* x, __global float* y, const int n)
{
    int i = get_global_id(0);
    if (i < n) {
        float xi = x[i];
        float c = 0.044715f * xi * xi * xi;
        float inner = 0.79788456f * (xi + c);
        float tanh_res = tanh(inner);
        y[i] = 0.5f * xi * (1.0f + tanh_res);
    }
}
)CLC";

std::vector<float> GeluOCL(const std::vector<float>& input, int platform_index) {
    cl_int err;
    size_t n = input.size();

    cl_uint num_platforms = 0;
    clGetPlatformIDs(0, nullptr, &num_platforms);
    if (num_platforms == 0)
        throw std::runtime_error("No OpenCL platforms found.");

    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

    if (platform_index >= (int)num_platforms)
        throw std::runtime_error("Invalid platform index.");

    cl_platform_id platform = platforms[platform_index];

    cl_uint num_devices = 0;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
    if (num_devices == 0)
        throw std::runtime_error("No GPU devices found on this platform.");

    std::vector<cl_device_id> devices(num_devices);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr);

    cl_device_id device = devices[0];

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS)
        throw std::runtime_error("Failed to create OpenCL context.");

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS)
        throw std::runtime_error("Failed to create command queue.");

    const char* src = gelu_kernel_src;
    size_t src_len = strlen(src);
    cl_program program = clCreateProgramWithSource(context, 1, &src, &src_len, &err);
    if (err != CL_SUCCESS)
        throw std::runtime_error("Failed to create program.");

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string log(log_size, '\0');
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::cerr << "OpenCL build log:\n" << log << std::endl;
        throw std::runtime_error("Failed to build program.");
    }

    cl_kernel kernel = clCreateKernel(program, "gelu_kernel", &err);
    if (err != CL_SUCCESS)
        throw std::runtime_error("Failed to create kernel.");

    size_t bytes = n * sizeof(float);
    cl_mem buf_in = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, (void*)input.data(), &err);
    cl_mem buf_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_in);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_out);
    clSetKernelArg(kernel, 2, sizeof(int), &n);

    size_t local_size = 256;
    size_t global_size = ((n + local_size - 1) / local_size) * local_size;

    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, &local_size, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
        throw std::runtime_error("Failed to enqueue kernel.");

    std::vector<float> output(n);
    clEnqueueReadBuffer(queue, buf_out, CL_TRUE, 0, bytes, output.data(), 0, nullptr, nullptr);

    clReleaseMemObject(buf_in);
    clReleaseMemObject(buf_out);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return output;
}
