#include "gelu_ocl.h"

#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#include <stdexcept>
#include <string>

static const char* GELU_KERNEL_SRC = R"CLC(
__kernel void gelu_kernel(__global const float* in,
                          __global float* out,
                          int n)
{
    int i = get_global_id(0);
    if (i >= n) return;

    float x  = in[i];
    float x2 = x * x;
    float x3 = x2 * x;

    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;

    float t = sqrt_2_over_pi * (x + coeff * x3);
    float e2t = exp(2.0f * t);
    float tanh_t = (e2t - 1.0f) / (e2t + 1.0f);

    out[i] = 0.5f * x * (1.0f + tanh_t);
}
)CLC";

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    const int n = static_cast<int>(input.size());
    if (n == 0) {
        return {};
    }

    cl_int err;

    cl_uint num_platforms = 0;
    err = clGetPlatformIDs(0, nullptr, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        throw std::runtime_error("GeluOCL: no OpenCL platforms");
    }

    std::vector<cl_platform_id> platforms(num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("GeluOCL: clGetPlatformIDs failed");
    }

    if (platform < 0 || static_cast<cl_uint>(platform) >= num_platforms) {
        throw std::runtime_error("GeluOCL: wrong platform index");
    }

    cl_platform_id plat = platforms[platform];

    cl_uint num_devices = 0;
    err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
    if (err != CL_SUCCESS || num_devices == 0) {
        throw std::runtime_error("GeluOCL: no GPU devices on platform");
    }

    std::vector<cl_device_id> devices(num_devices);
    err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("GeluOCL: clGetDeviceIDs failed");
    }

    cl_device_id device = devices[0];

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (!context || err != CL_SUCCESS) {
        throw std::runtime_error("GeluOCL: clCreateContext failed");
    }

    cl_command_queue queue =
        clCreateCommandQueue(context, device, 0, &err);
    if (!queue || err != CL_SUCCESS) {
        clReleaseContext(context);
        throw std::runtime_error("GeluOCL: clCreateCommandQueue failed");
    }

    const char* src = GELU_KERNEL_SRC;
    size_t src_len = std::strlen(GELU_KERNEL_SRC);
    cl_program program =
        clCreateProgramWithSource(context, 1, &src, &src_len, &err);
    if (!program || err != CL_SUCCESS) {
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error("GeluOCL: clCreateProgramWithSource failed");
    }

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string log(log_size, '\0');
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, &log[0], nullptr);

        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);

        throw std::runtime_error("GeluOCL: clBuildProgram failed: " + log);
    }

    cl_kernel kernel = clCreateKernel(program, "gelu_kernel", &err);
    if (!kernel || err != CL_SUCCESS) {
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error("GeluOCL: clCreateKernel failed");
    }

    cl_mem d_in = clCreateBuffer(context,
                                 CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 n * sizeof(float),
                                 const_cast<float*>(input.data()),
                                 &err);
    if (!d_in || err != CL_SUCCESS) {
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error("GeluOCL: clCreateBuffer d_in failed");
    }

    cl_mem d_out = clCreateBuffer(context,
                                  CL_MEM_WRITE_ONLY,
                                  n * sizeof(float),
                                  nullptr,
                                  &err);
    if (!d_out || err != CL_SUCCESS) {
        clReleaseMemObject(d_in);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error("GeluOCL: clCreateBuffer d_out failed");
    }

    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_in);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_out);
    err |= clSetKernelArg(kernel, 2, sizeof(int),    &n);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(d_out);
        clReleaseMemObject(d_in);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error("GeluOCL: clSetKernelArg failed");
    }

    size_t global = static_cast<size_t>(n);
    size_t local  = 256;
    if (global % local != 0) {
        global = ((global + local - 1) / local) * local;
    }

    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(d_out);
        clReleaseMemObject(d_in);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error("GeluOCL: clEnqueueNDRangeKernel failed");
    }

    std::vector<float> output(n);

    err = clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0,
                              n * sizeof(float), output.data(),
                              0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(d_out);
        clReleaseMemObject(d_in);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error("GeluOCL: clEnqueueReadBuffer failed");
    }

    clReleaseMemObject(d_out);
    clReleaseMemObject(d_in);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return output;
}
