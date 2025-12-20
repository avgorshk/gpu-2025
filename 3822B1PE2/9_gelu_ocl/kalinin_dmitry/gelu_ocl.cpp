#include "gelu_ocl.h"
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include <vector>
#include <iostream>
#include <cstring>

const char* gelu_kernel_optimized = R"(
__kernel void gelu_kernel_optimized(__global const float* input, 
                                    __global float* output, 
                                    int size) {
    int idx = get_global_id(0);
    int stride = get_global_size(0);
    
    for (int i = idx; i < size; i += stride) {
        float x = input[i];
        float x2 = x * x;
        float x3 = x2 * x;
        float inner = 0.7978845608f * (x + 0.044715f * x3);
        
        float exp_2z = exp(2.0f * inner);
        float inv_exp_2z_plus_one = 1.0f / (exp_2z + 1.0f);
        float tanh_val = 1.0f - 2.0f * inv_exp_2z_plus_one;
        
        output[i] = 0.5f * x * (1.0f + tanh_val);
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
static int cached_platform = -1;

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    size_t size = input.size();
    if (size == 0) return std::vector<float>();
    
    cl_int err;
    
    if (!initialized || cached_platform != platform) {
        if (initialized) {
            clReleaseKernel(kernel);
            clReleaseProgram(program);
            clReleaseCommandQueue(queue);
            clReleaseContext(context);
            if (d_input) clReleaseMemObject(d_input);
            if (d_output) clReleaseMemObject(d_output);
        }
        
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
        queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
        
        program = clCreateProgramWithSource(context, 1, &gelu_kernel_optimized, nullptr, &err);
        clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math", nullptr, nullptr);
        
        kernel = clCreateKernel(program, "gelu_kernel_optimized", &err);
        
        cached_platform = platform;
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
    
    cl_event write_event;
    clEnqueueWriteBuffer(queue, d_input, CL_FALSE, 0, bytes, input.data(), 0, nullptr, &write_event);
    
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output);
    int size_int = static_cast<int>(size);
    clSetKernelArg(kernel, 2, sizeof(int), &size_int);
    
    size_t global_size = size;
    size_t local_size = 256;
    if (global_size % local_size != 0) {
        global_size = ((global_size + local_size - 1) / local_size) * local_size;
    }
    cl_event kernel_event;
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, &local_size, 1, &write_event, &kernel_event);
    
    std::vector<float> output(size);
    clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, bytes, output.data(), 1, &kernel_event, nullptr);
    
    clFinish(queue);
    
    return output;
}

