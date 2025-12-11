#include "gelu_ocl.h"
#include <CL/opencl.h>
#include <cmath>
#include <string>

const char* geluKernelSrc = R"(
__kernel void gelu_calc(__global const float* in_vals, __global float* out_vals, int len) {
    int idx = get_global_id(0);
    if (idx < len) {
        float x = in_vals[idx];
        const float sqrt2pi = 0.7978845608f;
        const float coef = 0.044715f;
        float x3 = x * x * x;
        float arg = sqrt2pi * (x + coef * x3);
        float exp_term = exp(2.0f * arg);
        float tanh_term = (exp_term - 1.0f) / (exp_term + 1.0f);
        out_vals[idx] = 0.5f * x * (1.0f + tanh_term);
    }
}
)";

std::vector<float> GeluOCL(const std::vector<float>& input, int platformIdx) {
    int n = input.size();
    if (n == 0) return {};

    std::vector<float> output(n);

    cl_platform_id platId;
    cl_device_id devId;
    cl_uint numPlats, numDevs;

    clGetPlatformIDs(0, nullptr, &numPlats);
    cl_platform_id* plats = new cl_platform_id[numPlats];
    clGetPlatformIDs(numPlats, plats, nullptr);
    platId = plats[platformIdx];
    delete[] plats;

    clGetDeviceIDs(platId, CL_DEVICE_TYPE_GPU, 1, &devId, &numDevs);

    cl_int err;
    cl_context ctx = clCreateContext(nullptr, 1, &devId, nullptr, nullptr, &err);
    cl_command_queue queue = clCreateCommandQueue(ctx, devId, 0, &err);

    size_t bytes = n * sizeof(float);
    cl_mem buf_in = clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytes, nullptr, &err);
    cl_mem buf_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);

    clEnqueueWriteBuffer(queue, buf_in, CL_TRUE, 0, bytes, input.data(), 0, nullptr, nullptr);

    cl_program prog = clCreateProgramWithSource(ctx, 1, &geluKernelSrc, nullptr, &err);
    clBuildProgram(prog, 1, &devId, nullptr, nullptr, nullptr);

    cl_kernel kernel = clCreateKernel(prog, "gelu_calc", &err);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_in);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_out);
    clSetKernelArg(kernel, 2, sizeof(int), &n);

    size_t local = 256;
    size_t global = (n + local - 1) / local * local;
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr);

    clEnqueueReadBuffer(queue, buf_out, CL_TRUE, 0, bytes, output.data(), 0, nullptr, nullptr);

    clReleaseKernel(kernel);
    clReleaseProgram(prog);
    clReleaseMemObject(buf_in);
    clReleaseMemObject(buf_out);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    return output;
}
