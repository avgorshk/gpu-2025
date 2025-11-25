#include "gelu_ocl.h"
#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <CL/opencl.hpp>
#include <string>
#include <stdexcept>

namespace {
    const char* geluKernelSource = R"CLC(
        __kernel void gelu(
            __global const float* input,
            __global float* output,
            const int count)
        {
            int i = get_global_id(0);
            if (i < count) {
                float x = input[i];
                float x_cubed = x * x * x;
                float approx = 1.0f + exp(-1.59577f * (x + 0.044715f * x_cubed));
                output[i] = x / approx;
            }
        }
    )CLC";
}

std::vector<float> GeluOCL(const std::vector<float>& input) {
    if (input.empty()) {
        return {};
    }

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        throw std::runtime_error("OpenCL: No platforms found.");
    }

    cl::Platform platform = platforms.front();

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    if (devices.empty()) {
        throw std::runtime_error("OpenCL: No devices found.");
    }

    cl::Device device = devices.front();
    cl::Context context(device);
    cl::CommandQueue queue(context, device);

    cl::Program::Sources sources;
    sources.push_back({ geluKernelSource, strlen(geluKernelSource) });

    cl::Program program(context, sources);
    if (program.build({ device }) != CL_SUCCESS) {
        std::string build_log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
        throw std::runtime_error("OpenCL build failed:\n" + build_log);
    }

    cl::Kernel kernel(program, "gelu");

    size_t size = input.size();
    size_t bytes = size * sizeof(float);
    cl::Buffer bufferIn(context, CL_MEM_READ_ONLY, bytes);
    cl::Buffer bufferOut(context, CL_MEM_WRITE_ONLY, bytes);

    queue.enqueueWriteBuffer(bufferIn, CL_TRUE, 0, bytes, input.data());

    kernel.setArg(0, bufferIn);
    kernel.setArg(1, bufferOut);
    kernel.setArg(2, static_cast<int>(size));

    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(size), cl::NullRange);

    std::vector<float> result(size);
    queue.enqueueReadBuffer(bufferOut, CL_TRUE, 0, bytes, result.data());

    return result;
}