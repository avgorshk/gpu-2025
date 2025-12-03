#include "gelu_ocl.h"
#include <vector>
#include <CL/cl.h>
#include <string>
#include <iostream>
#include <cmath>

static cl_context context = nullptr;
static cl_command_queue queue = nullptr;
static cl_program program = nullptr;
static cl_kernel kernel = nullptr;
static bool is_initialized = false;

const char* gelu_kernel_source = R"(
__kernel void gelu_kernel(__global const float* input, __global float* output, int size) {
    int idx = get_global_id(0);
    
    if (idx < size) {
        const float sqrt_2_over_pi = 0.7978845608028654f;
        const float coeff = 0.044715f;
        const float c_half = 0.5f; 
        
        float x = input[idx];
        float x_cubed = x * x * x;
        
        float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
        float tanh_val = tanh(inner); 
        
        output[idx] = c_half * x * (1.0f + tanh_val);
    }
}
)";

void initOpenCL(int platform_idx) {
    if (is_initialized) return;

    cl_int err;
    cl_uint num_platforms;
    clGetPlatformIDs(0, nullptr, &num_platforms);
    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

    if (platform_idx < 0 || platform_idx >= num_platforms) {
        platform_idx = 0;
    }
    cl_platform_id platform_id = platforms[platform_idx];

    cl_device_id device_id;
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, nullptr);
    if (err != CL_SUCCESS) {
        clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 1, &device_id, nullptr);
    }

    context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &err);

#ifdef CL_VERSION_2_0
    queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
#else
    queue = clCreateCommandQueue(context, device_id, 0, &err);
#endif

    program = clCreateProgramWithSource(context, 1, &gelu_kernel_source, nullptr, &err);
    err = clBuildProgram(program, 1, &device_id, nullptr, nullptr, nullptr);

    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::cerr << "Build error: " << log.data() << std::endl;
    }

    kernel = clCreateKernel(program, "gelu_kernel", &err);

    is_initialized = true;
}

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    initOpenCL(platform);

    size_t size = input.size();
    if (size == 0) return {};

    std::vector<float> output(size);
    size_t byte_size = size * sizeof(float);
    cl_int err;

    cl_mem d_input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, byte_size, (void*)input.data(), &err);
    cl_mem d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, byte_size, nullptr, &err);

    int size_arg = static_cast<int>(size);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output);
    clSetKernelArg(kernel, 2, sizeof(int), &size_arg);

    size_t local_size = 256;
    size_t global_size = ((size + local_size - 1) / local_size) * local_size;

    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, &local_size, 0, nullptr, nullptr);

    err = clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, byte_size, output.data(), 0, nullptr, nullptr);

    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);

    return output;
}