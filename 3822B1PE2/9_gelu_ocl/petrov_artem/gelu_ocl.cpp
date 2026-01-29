#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include "gelu_ocl.h"
#include <CL/cl.h>
#include <vector>
#include <cstring>

std::vector<float> GeluOCL(const std::vector<float>& input, int platform_id) {
    if (input.empty()) {
        return {};
    }

    size_t n = input.size();
    cl_int err;

    cl_uint num_platforms = 0;
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        return std::vector<float>(n, 0.0f);
    }

    std::vector<cl_platform_id> platforms(num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms.data(), NULL);
    if (err != CL_SUCCESS) {
        return std::vector<float>(n, 0.0f);
    }

    if (platform_id < 0 || static_cast<cl_uint>(platform_id) >= num_platforms) {
        return std::vector<float>(n, 0.0f);
    }
    cl_platform_id platform = platforms[platform_id];

    cl_uint num_devices = 0;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
    if (err != CL_SUCCESS || num_devices == 0) {
        return std::vector<float>(n, 0.0f);
    }

    std::vector<cl_device_id> devices(num_devices);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), NULL);
    if (err != CL_SUCCESS || devices.empty()) {
        return std::vector<float>(n, 0.0f);
    }
    cl_device_id device = devices[0];

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        return std::vector<float>(n, 0.0f);
    }

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS) {
        clReleaseContext(context);
        return std::vector<float>(n, 0.0f);
    }

    const char* kernel_source =
        "__kernel void gelu_ocl(__global const float* input,\n"
        "                       __global float* output,\n"
        "                       const uint n) {\n"
        "    const uint idx = get_global_id(0);\n"
        "    if (idx >= n) return;\n"
        "    const float x = input[idx];\n"
        "    const float x3 = x * x * x;\n"
        "    const float SQRT_2_OVER_PI = 0.7978845608f;\n"
        "    const float GELU_COEFF = 0.044715f;\n"
        "    const float MUL = 2.0f * SQRT_2_OVER_PI;\n"
        "    const float z = MUL * (x + GELU_COEFF * x3);\n"
        "    output[idx] = x / (1.0f + native_exp(-z));\n"
        "}";

    cl_program program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &err);
    if (err != CL_SUCCESS) {
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return std::vector<float>(n, 0.0f);
    }

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return std::vector<float>(n, 0.0f);
    }

    cl_kernel kernel = clCreateKernel(program, "gelu_ocl", &err);
    if (err != CL_SUCCESS) {
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return std::vector<float>(n, 0.0f);
    }

    size_t bytes = n * sizeof(float);
    cl_mem input_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        bytes, const_cast<float*>(input.data()), &err);
    if (err != CL_SUCCESS) {
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return std::vector<float>(n, 0.0f);
    }

    cl_mem output_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, &err);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(input_buf);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return std::vector<float>(n, 0.0f);
    }

    cl_uint n_uint = static_cast<cl_uint>(n);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buf);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buf);
    clSetKernelArg(kernel, 2, sizeof(cl_uint), &n_uint);

    size_t global_size = ((n + 255) / 256) * 256;
    size_t local_size = 256;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(output_buf);
        clReleaseMemObject(input_buf);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return std::vector<float>(n, 0.0f);
    }

    clFinish(queue);

    std::vector<float> output(n);
    err = clEnqueueReadBuffer(queue, output_buf, CL_TRUE, 0, bytes, output.data(), 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        output.assign(n, 0.0f);
    }

    clReleaseMemObject(output_buf);
    clReleaseMemObject(input_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return output;
}