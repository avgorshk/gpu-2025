#include "gelu_ocl.h"

#include <CL/cl.h>
#include <iostream>
#include <algorithm>

const char* geluKernelSource = R"(
__kernel void gelu(__global const float* input,
                   __global float* output,
                   const int size)
{
    int id = get_global_id(0);
    if (id < size) {
        float x = input[id];
        float t = 0.7978845608f * (x + 0.044715f * x * x * x);
        output[id] = 0.5f * x * (1.0f + tanh(t));
    }
}
)";

std::vector<float> GeluOCL(const std::vector<float>& input, int platformIndex) {
    if (input.empty()) return {};

    cl_int err;

    cl_uint numPlatforms = 0;
    clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (numPlatforms == 0) return {};

    std::vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);

    if (platformIndex < 0 || platformIndex >= (int)numPlatforms) return {};
    cl_platform_id platform = platforms[platformIndex];

    cl_uint numDevices = 0;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
    cl_device_type dtype = CL_DEVICE_TYPE_GPU;

    if (err != CL_SUCCESS || numDevices == 0) {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, nullptr, &numDevices);
        dtype = CL_DEVICE_TYPE_CPU;
        if (err != CL_SUCCESS || numDevices == 0) return {};
    }

    std::vector<cl_device_id> devices(numDevices);
    clGetDeviceIDs(platform, dtype, numDevices, devices.data(), nullptr);
    cl_device_id device = devices[0];

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) return {};

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS) return {};

    cl_program program =
        clCreateProgramWithSource(context, 1, &geluKernelSource, nullptr, &err);
    if (err != CL_SUCCESS) return {};

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logSize = 0;
        clGetProgramBuildInfo(program, device,
                              CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::vector<char> log(logSize);
        clGetProgramBuildInfo(program, device,
                              CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        std::cerr << "Build log:\n" << log.data() << std::endl;
        return {};
    }

    cl_kernel kernel = clCreateKernel(program, "gelu", &err);
    if (err != CL_SUCCESS) return {};

    size_t bytes = input.size() * sizeof(float);
    cl_mem d_input = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        bytes, (void*)input.data(), &err);

    cl_mem d_output = clCreateBuffer(
        context, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);

    int n = (int)input.size();
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output);
    clSetKernelArg(kernel, 2, sizeof(int), &n);

    size_t maxWG = 0;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                    sizeof(size_t), &maxWG, nullptr);

    size_t localSize = std::min((size_t)256, maxWG);
    size_t globalSize =
        ((input.size() + localSize - 1) / localSize) * localSize;

    err = clEnqueueNDRangeKernel(
        queue, kernel, 1, nullptr,
        &globalSize, &localSize, 0, nullptr, nullptr);

    if (err != CL_SUCCESS) return {};

    std::vector<float> output(input.size());
    clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0,
                        bytes, output.data(), 0, nullptr, nullptr);

    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return output;
}