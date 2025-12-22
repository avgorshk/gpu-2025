#include "gelu_ocl.h"
#include <vector>
#include <CL/cl.h>
#include <string>
#include <stdexcept>

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    const std::string kernel_source = R"(
        __kernel void gelu_kernel(__global const float* input,
                                 __global float* output,
                                 const int size) {
            int idx = get_global_id(0);
            if (idx < size) {
                float x = input[idx];
                float x3 = x * x * x;
                float inner = 0.7978845608028654f * (x + 0.044715f * x3);
                float tanh_val = tanh(inner);
                output[idx] = 0.5f * x * (1.0f + tanh_val);
            }
        }
    )";
    
    cl_platform_id platforms[10];
    cl_uint num_platforms;
    clGetPlatformIDs(10, platforms, &num_platforms);
    
    if (platform >= (int)num_platforms) {
        throw std::runtime_error("Invalid platform index");
    }
    
    cl_device_id device;
    clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, nullptr, nullptr);
    
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, nullptr, nullptr);
    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    
    cl_kernel kernel = clCreateKernel(program, "gelu_kernel", nullptr);
    
    size_t size = input.size();
    std::vector<float> output(size);
    
    cl_mem d_input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    size * sizeof(float), (void*)input.data(), nullptr);
    cl_mem d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                     size * sizeof(float), nullptr, nullptr);
    
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output);
    clSetKernelArg(kernel, 2, sizeof(int), &size);
    
    size_t global_size = size;
    size_t local_size = 256;
    
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, &local_size, 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, size * sizeof(float), output.data(), 0, nullptr, nullptr);
    
    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    return output;
}