#include "gelu_ocl.h"
#include <CL/cl.h>
#include <cmath>
#include <iostream>
#include <stdexcept>

const char* kernelSource = R"CLC(
__kernel void gelu_kernel(__global const float* input,
                          __global float* output,
                          const int n)
{
    int i = get_global_id(0);
    if (i < n) {
        float x = input[i];
        // Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float c = 0.79788456f * (x + 0.044715f * x * x * x);
        float y = 0.5f * x * (1.0f + tanh(c));
        output[i] = y;
    }
}
)CLC";

std::vector<float> GeluOCL(const std::vector<float>& input, int platformIndex) {
    if (input.empty()) return {};

    cl_int err;
    cl_uint numPlatforms = 0;
    clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (numPlatforms == 0) throw std::runtime_error("No OpenCL platforms found");
    if (platformIndex >= static_cast<int>(numPlatforms)) throw std::runtime_error("Invalid platform index");

    std::vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    cl_platform_id platform = platforms[platformIndex];

    cl_uint numDevices = 0;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
    if (numDevices == 0) throw std::runtime_error("No GPU devices found on this platform");

    std::vector<cl_device_id> devices(numDevices);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr);
    cl_device_id device = devices[0];

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create OpenCL context");

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create command queue");

    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create OpenCL program");

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logSize = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::vector<char> log(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        std::cerr << "Build log:\n" << log.data() << std::endl;
        throw std::runtime_error("Failed to build OpenCL program");
    }

    cl_kernel kernel = clCreateKernel(program, "gelu_kernel", &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create kernel");

    const size_t bytes = input.size() * sizeof(float);
    cl_mem d_input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, (void*)input.data(), &err);
    cl_mem d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);

    int n = static_cast<int>(input.size());
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output);
    clSetKernelArg(kernel, 2, sizeof(int), &n);

    size_t globalSize = ((n + 255) / 256) * 256;
    size_t localSize = 256;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, &localSize, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to enqueue kernel");

    std::vector<float> output(input.size());
    clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, bytes, output.data(), 0, nullptr, nullptr);

    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return output;
}