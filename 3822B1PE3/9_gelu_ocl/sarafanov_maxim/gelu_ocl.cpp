#include "gelu_ocl.h"
#include <CL/cl.h>
#include <vector>
#include <iostream>
#include <string>
#include <cmath>

#define CHECK_OPENCL_ERROR(call) \
    do { \
        cl_int err = call; \
        if (err != CL_SUCCESS) { \
            std::cerr << "OpenCL error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << err << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

const std::string GELU_KERNEL = R"(
__kernel void gelu(__global const float* input, __global float* output, int n) {
    int idx = get_global_id(0);
    if (idx >= n) return;
    
    float x = input[idx];
    float cdf = 0.5f * (1.0f + tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
    output[idx] = x * cdf;
}
)";

const std::string GELU_APPROX_KERNEL = R"(
__kernel void gelu_approx(__global const float* input, __global float* output, int n) {
    int idx = get_global_id(0);
    if (idx >= n) return;
    
    float x = input[idx];
    float x3 = x * x * x;
    float inner = 0.044715f * x3;
    float cdf = 0.5f * (1.0f + tanh(0.7978845608028654f * (x + inner)));
    output[idx] = x * cdf;
}
)";

const std::string GELU_FAST_KERNEL = R"(
__kernel void gelu_fast(__global const float* input, __global float* output, int n) {
    int idx = get_global_id(0);
    if (idx >= n) return;
    
    float x = input[idx];
    float x3 = x * x * x;
    float inner = 0.044715f * x3;
    float temp = 0.7978845608028654f * (x + inner);
    float cdf = 0.5f * (1.0f + (1.0f - exp(-2.0f * temp)) / (1.0f + exp(-2.0f * temp)));
    output[idx] = x * cdf;
}
)";

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    cl_uint num_platforms;
    CHECK_OPENCL_ERROR(clGetPlatformIDs(0, NULL, &num_platforms));
    
    if (num_platforms == 0) {
        throw std::runtime_error("No OpenCL platforms found");
    }
    
    std::vector<cl_platform_id> platforms(num_platforms);
    CHECK_OPENCL_ERROR(clGetPlatformIDs(num_platforms, platforms.data(), NULL));
    
    if (platform >= num_platforms) {
        throw std::runtime_error("Platform index out of range");
    }
    
    cl_platform_id selected_platform = platforms[platform];
    
    cl_uint num_devices;
    CHECK_OPENCL_ERROR(clGetDeviceIDs(selected_platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices));
    
    if (num_devices == 0) {
        throw std::runtime_error("No GPU devices found on selected platform");
    }
    
    std::vector<cl_device_id> devices(num_devices);
    CHECK_OPENCL_ERROR(clGetDeviceIDs(selected_platform, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), NULL));
    
    cl_device_id device = devices[0];
    
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);
    
    const char* kernel_source = GELU_FAST_KERNEL.c_str();
    cl_program program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, NULL);
    
    cl_int build_err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (build_err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), NULL);
        std::cerr << "Build error: " << log.data() << std::endl;
        throw std::runtime_error("OpenCL program build failed");
    }
    
    cl_kernel kernel = clCreateKernel(program, "gelu_fast", NULL);
    
    size_t n = input.size();
    size_t buffer_size = n * sizeof(float);
    
    cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buffer_size, (void*)input.data(), NULL);
    cl_mem output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, buffer_size, NULL, NULL);
    
    CHECK_OPENCL_ERROR(clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer));
    CHECK_OPENCL_ERROR(clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buffer));
    CHECK_OPENCL_ERROR(clSetKernelArg(kernel, 2, sizeof(int), &n));
    
    size_t global_size = n;
    size_t local_size = 256;
    
    CHECK_OPENCL_ERROR(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL));
    
    std::vector<float> output(n);
    CHECK_OPENCL_ERROR(clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, buffer_size, output.data(), 0, NULL, NULL));
    
    CHECK_OPENCL_ERROR(clFinish(queue));
    
    CHECK_OPENCL_ERROR(clReleaseMemObject(input_buffer));
    CHECK_OPENCL_ERROR(clReleaseMemObject(output_buffer));
    CHECK_OPENCL_ERROR(clReleaseKernel(kernel));
    CHECK_OPENCL_ERROR(clReleaseProgram(program));
    CHECK_OPENCL_ERROR(clReleaseCommandQueue(queue));
    CHECK_OPENCL_ERROR(clReleaseContext(context));
    
    return output;
}