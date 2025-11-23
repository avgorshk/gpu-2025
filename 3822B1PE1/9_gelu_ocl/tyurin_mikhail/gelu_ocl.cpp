#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include "gelu_ocl.h"
#include <cstring>
#include <CL/cl.h>
#include <vector>
#include <string>
#include <chrono>
#include <iostream>

static const char* gelu_kernel_source = R"CLC(
__kernel void gelu_kernel(__global const float* in, __global float* out, const unsigned int n_total) {
    const unsigned int idx = get_global_id(0);
    if (idx >= n_total) return;
    const float k0 = 0.7978845608028654f;
    const float k1 = 0.044715f;
    float x = in[idx];
    float x3 = x * x * x;
    float inner = k0 * (x + k1 * x3);
    float e = exp(2.0f * inner);
    float t = (e - 1.0f) / (e + 1.0f);
    out[idx] = 0.5f * x * (1.0f + t);
}
)CLC";

std::vector<float> GeluOCL(const std::vector<float>& input, int platform_index) {
    size_t n_elems = input.size();
    if (n_elems == 0) return {};

    cl_uint num_platforms = 0;
    clGetPlatformIDs(0, nullptr, &num_platforms);
    if (num_platforms == 0) return {};

    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

    if (platform_index < 0 || (cl_uint)platform_index >= num_platforms) platform_index = 0;
    cl_platform_id platform = platforms[platform_index];

    cl_uint num_devices = 0;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
    if (num_devices == 0) return {};

    std::vector<cl_device_id> devices(num_devices);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr);
    cl_device_id device = devices[0];

    cl_int err;
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (!context) return {};

    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
    if (!queue) {
        clReleaseContext(context);
        return {};
    }

    const char* src = gelu_kernel_source;
    size_t src_len = strlen(src);
    cl_program program = clCreateProgramWithSource(context, 1, &src, &src_len, &err);
    if (!program) {
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return {};
    }

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        if (log_size) {
            std::string log;
            log.resize(log_size);
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, &log[0], nullptr);
            std::cerr << log << std::endl;
        }
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return {};
    }

    cl_kernel kernel = clCreateKernel(program, "gelu_kernel", &err);
    if (!kernel) {
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return {};
    }

    cl_mem in_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   n_elems * sizeof(float), const_cast<float*>(input.data()), &err);
    cl_mem out_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                    n_elems * sizeof(float), nullptr, &err);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_buf);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_buf);
    unsigned int total = static_cast<unsigned int>(n_elems);
    clSetKernelArg(kernel, 2, sizeof(unsigned int), &total);

    size_t local = 256;
    size_t global = ((n_elems + local - 1) / local) * local;

    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr);
    clFinish(queue);

    std::vector<float> out(n_elems);
    clEnqueueReadBuffer(queue, out_buf, CL_TRUE, 0, n_elems * sizeof(float), out.data(), 0, nullptr, nullptr);

    clReleaseMemObject(in_buf);
    clReleaseMemObject(out_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return out;
}
