#include "gelu_ocl.h"
#include <vector>
#include <CL/cl.h>
#include <string>

const char* gelu_kernel_source = R"(
__kernel void gelu_kernel(__global const float* input, __global float* output, int size) {
    int idx = get_global_id(0);
    
    if (idx < size) {
        const float sqrt_2_over_pi = 0.7978845608028654f;
        const float coeff = 0.044715f;
        const float half = 0.5f;
        
        float x = input[idx];
        float x_cubed = x * x * x;
        
        float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
        float exp_val = exp(-2.0f * inner);
        float tanh_val = (1.0f - exp_val) / (1.0f + exp_val);
        
        output[idx] = half * x * (1.0f + tanh_val);
    }
}
)";

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    const size_t size = input.size();
    std::vector<float> output(size);
    
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem d_input, d_output;
    
    clGetPlatformIDs(1, &platform_id, nullptr);
    clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, nullptr);
    
    context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, nullptr);
    queue = clCreateCommandQueue(context, device_id, 0, nullptr);
    
    const size_t byte_size = size * sizeof(float);
    d_input = clCreateBuffer(context, CL_MEM_READ_ONLY, byte_size, nullptr, nullptr);
    d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, byte_size, nullptr, nullptr);
    
    clEnqueueWriteBuffer(queue, d_input, CL_TRUE, 0, byte_size, input.data(), 0, nullptr, nullptr);
    
    program = clCreateProgramWithSource(context, 1, &gelu_kernel_source, nullptr, nullptr);
    clBuildProgram(program, 1, &device_id, nullptr, nullptr, nullptr);
    
    kernel = clCreateKernel(program, "gelu_kernel", nullptr);
    
    int size_arg = static_cast<int>(size);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output);
    clSetKernelArg(kernel, 2, sizeof(int), &size_arg);
    
    size_t global_size = size;
    size_t local_size = 256;
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, &local_size, 0, nullptr, nullptr);
    
    clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, byte_size, output.data(), 0, nullptr, nullptr);
    
    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    return output;
}