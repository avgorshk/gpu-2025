#include "gelu_ocl.h"
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include <vector>
#include <iostream>
#include <cstring>

const char* gelu_kernel_source = R"(
__kernel void gelu_kernel(__global const float* input, __global float* output, int size) {
    int idx = get_global_id(0);
    if (idx < size) {
        float x = input[idx];
        float x3 = x * x * x;
        float inner = 0.7978845608f * (x + 0.044715f * x3);
        
        float exp_2z = exp(2.0f * inner);
        float tanh_val = (exp_2z - 1.0f) / (exp_2z + 1.0f);
        
        output[idx] = 0.5f * x * (1.0f + tanh_val);
    }
}
)";

static cl_context context = nullptr;
static cl_command_queue queue = nullptr;
static cl_program program = nullptr;
static cl_kernel kernel = nullptr;
static cl_mem d_input = nullptr;
static cl_mem d_output = nullptr;
static size_t allocated_size = 0;
static bool initialized = false;

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    size_t size = input.size();
    if (size == 0) return std::vector<float>();
    
    cl_int err;
    
    if (!initialized) {
        cl_uint num_platforms;
        clGetPlatformIDs(0, nullptr, &num_platforms);
        
        std::vector<cl_platform_id> platforms(num_platforms);
        clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
        
        if (platform >= static_cast<int>(num_platforms)) {
            std::cerr << "Invalid platform index" << std::endl;
            return std::vector<float>();
        }
        
        cl_uint num_devices;
        clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
        
        if (num_devices == 0) {
            std::cerr << "No GPU devices found" << std::endl;
            return std::vector<float>();
        }
        
        cl_device_id device;
        clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
        
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        queue = clCreateCommandQueue(context, device, 0, &err);
        
        program = clCreateProgramWithSource(context, 1, &gelu_kernel_source, nullptr, &err);
        clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        
        kernel = clCreateKernel(program, "gelu_kernel", &err);
        
        initialized = true;
    }
    
    size_t bytes = size * sizeof(float);
    
    if (d_input == nullptr || allocated_size < size) {
        if (d_input != nullptr) {
            clReleaseMemObject(d_input);
            clReleaseMemObject(d_output);
        }
        d_input = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, nullptr, &err);
        d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);
        allocated_size = size;
    }
    
    clEnqueueWriteBuffer(queue, d_input, CL_TRUE, 0, bytes, input.data(), 0, nullptr, nullptr);
    
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output);
    clSetKernelArg(kernel, 2, sizeof(int), &size);
    
    size_t global_size = size;
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr);
    
    std::vector<float> output(size);
    clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, bytes, output.data(), 0, nullptr, nullptr);
    
    clFinish(queue);
    
    return output;
}

