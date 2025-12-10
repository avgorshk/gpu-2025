#include "gelu_ocl.h"
#include <CL/cl.h>
#include <cmath>
#include <iostream>
#include <stdexcept>

const char* oclGeluKernelSrc = R"CLC(
__kernel void act_gelu(__global const float* src,
                       __global float* dst,
                       const int count)
{
    int id = get_global_id(0);
    if (id < count) {
        float v = src[id];
        float t = 0.79788456f * (v + 0.044715f * v * v * v);
        float r = 0.5f * v * (1.0f + tanh(t));
        dst[id] = r;
    }
}
)CLC";

std::vector<float> runGeluOpenCL(const std::vector<float>& data, int platformIdx) {
    if (data.empty()) return {};

    cl_int status;

    // ----- PLATFORM SELECT -----
    cl_uint platCount = 0;
    clGetPlatformIDs(0, nullptr, &platCount);
    if (platCount == 0) throw std::runtime_error("OpenCL: no platforms available");

    if (platformIdx >= (int)platCount)
        throw std::runtime_error("OpenCL: platform index out of range");

    std::vector<cl_platform_id> platList(platCount);
    clGetPlatformIDs(platCount, platList.data(), nullptr);
    cl_platform_id platform = platList[platformIdx];

    // ----- DEVICE SELECT -----
    cl_uint devCount = 0;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &devCount);
    if (devCount == 0) throw std::runtime_error("OpenCL: no GPU devices found");

    std::vector<cl_device_id> deviceList(devCount);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, devCount, deviceList.data(), nullptr);
    cl_device_id device = deviceList[0];

    // ----- CONTEXT & QUEUE -----
    cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status);
    if (status != CL_SUCCESS) throw std::runtime_error("OpenCL: failed to create context");

    cl_command_queue queue = clCreateCommandQueue(ctx, device, 0, &status);
    if (status != CL_SUCCESS) throw std::runtime_error("OpenCL: failed to create command queue");

    // ----- PROGRAM -----
    cl_program program = clCreateProgramWithSource(ctx, 1, &oclGeluKernelSrc, nullptr, &status);
    if (status != CL_SUCCESS) throw std::runtime_error("OpenCL: program creation failed");

    status = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (status != CL_SUCCESS) {
        size_t logSz = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSz);
        std::vector<char> log(logSz);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSz, log.data(), nullptr);
        std::cerr << "OpenCL build log:\n" << log.data() << "\n";
        throw std::runtime_error("OpenCL: build error");
    }

    cl_kernel kernel = clCreateKernel(program, "act_gelu", &status);
    if (status != CL_SUCCESS) throw std::runtime_error("OpenCL: kernel creation failed");

    // ----- BUFFERS -----
    size_t byteCount = data.size() * sizeof(float);
    cl_mem bufIn = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  byteCount, (void*)data.data(), &status);
    cl_mem bufOut = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, byteCount, nullptr, &status);

    int count = (int)data.size();
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufIn);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufOut);
    clSetKernelArg(kernel, 2, sizeof(int), &count);

    // ----- DISPATCH -----
    size_t local = 256;
    size_t global = ((count + local - 1) / local) * local;

    status = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr,
                                    &global, &local, 0, nullptr, nullptr);
    if (status != CL_SUCCESS) throw std::runtime_error("OpenCL: kernel enqueue failed");

    // ----- RESULT -----
    std::vector<float> result(data.size());
    clEnqueueReadBuffer(queue, bufOut, CL_TRUE, 0, byteCount, result.data(),
                        0, nullptr, nullptr);

    // ----- CLEANUP -----
    clReleaseMemObject(bufIn);
    clReleaseMemObject(bufOut);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    return result;
}
