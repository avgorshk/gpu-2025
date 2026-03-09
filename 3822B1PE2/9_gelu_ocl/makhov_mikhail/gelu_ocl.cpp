#include "gelu_ocl.h"
#include <CL/cl.h>
#include <vector>

const char* gelu_kernel_source = R"(
__kernel void compute_gelu(__global const float* input, __global float* output, int count) {
    int i = get_global_id(0);
    if (i < count) {
        float x = input[i];
        const float sqrt_2_over_pi = 0.7978845608028654f;
        const float coeff = 0.044715f;
        float x2 = x * x;
        float x3 = x2 * x;
        float inner = sqrt_2_over_pi * (x + coeff * x3);
        
        float exp_2z = exp(2.0f * inner);
        float tanh_val = (exp_2z - 1.0f) / (exp_2z + 1.0f);
        
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
static int cached_platform = -1;
static size_t allocated_size = 0;
static bool initialized = false;

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    size_t size = input.size();
    if (size == 0) {
        return std::vector<float>();
    }
    
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
        
        cl_device_id device;
        clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
        
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
        queue = clCreateCommandQueue(context, device, 0, nullptr);
        
        program = clCreateProgramWithSource(context, 1, &gelu_kernel_source, nullptr, nullptr);
        clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        
        kernel = clCreateKernel(program, "compute_gelu", nullptr);
        
        cached_platform = platform;
        initialized = true;
    }
    
    size_t bytes = size * sizeof(float);
    
    if (d_input == nullptr || allocated_size < size) {
        if (d_input != nullptr) {
            clReleaseMemObject(d_input);
            clReleaseMemObject(d_output);
        }
        d_input = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, nullptr, nullptr);
        d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, nullptr);
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
    
    cl_event read_event;
    clEnqueueReadBuffer(queue, d_output, CL_FALSE, 0, bytes, output.data(), 1, &kernel_event, &read_event);
    
    clWaitForEvents(1, &read_event);
    
    clReleaseEvent(read_event);
    clReleaseEvent(kernel_event);
    clReleaseEvent(write_event);
    
    return output;
}

