#define CL_TARGET_OPENCL_VERSION 300

#include "gelu_ocl.h"
#include <CL/cl.h>
#include <cstring>
#include <iostream>
#include <stdexcept>

static const char* gelu_opencl_kernel = R"(
__kernel void compute_gelu(__global const float* in_data, 
                           __global float* out_data, 
                           const unsigned int count) {
    const unsigned int gid = get_global_id(0);
    if (gid >= count) return;
    
    const float sqrt_2_over_pi = 0.7978845608f;
    const float cubic_coeff = 0.044715f;
    
    float val = in_data[gid];
    float val_cubed = val * val * val;
    float inner_term = sqrt_2_over_pi * (val + cubic_coeff * val_cubed);
    
    float e2x = exp(2.0f * inner_term);
    float tanh_result = (e2x - 1.0f) / (e2x + 1.0f);
    
    out_data[gid] = 0.5f * val * (1.0f + tanh_result);
}
)";

static struct {
    cl_platform_id plat;
    cl_device_id dev;
    cl_context ctx;
    cl_command_queue cmd_queue;
    cl_program prog;
    cl_kernel kern;
    cl_mem mem_in;
    cl_mem mem_out;
    size_t buffer_capacity;
    bool ready;
} ocl_state = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, false};

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    if (input.empty()) return {};
    
    size_t element_count = input.size();
    std::vector<float> result(element_count);
    cl_int status;
    
    if (!ocl_state.ready) {
        cl_uint platform_count = 0;
        clGetPlatformIDs(0, nullptr, &platform_count);
        if (platform_count == 0) throw std::runtime_error("No OpenCL platforms found");
        
        std::vector<cl_platform_id> available_platforms(platform_count);
        clGetPlatformIDs(platform_count, available_platforms.data(), nullptr);
        
        int selected_platform = (platform >= 0 && platform < (int)platform_count) ? platform : 0;
        ocl_state.plat = available_platforms[selected_platform];
        
        cl_uint device_count = 0;
        clGetDeviceIDs(ocl_state.plat, CL_DEVICE_TYPE_GPU, 0, nullptr, &device_count);
        if (device_count == 0) throw std::runtime_error("No GPU devices found");
        
        std::vector<cl_device_id> gpu_devices(device_count);
        clGetDeviceIDs(ocl_state.plat, CL_DEVICE_TYPE_GPU, device_count, gpu_devices.data(), nullptr);
        ocl_state.dev = gpu_devices[0];
        
        ocl_state.ctx = clCreateContext(nullptr, 1, &ocl_state.dev, nullptr, nullptr, &status);
        if (status != CL_SUCCESS) throw std::runtime_error("Failed to create context");
        
        ocl_state.cmd_queue = clCreateCommandQueueWithProperties(ocl_state.ctx, ocl_state.dev, 0, &status);
        if (status != CL_SUCCESS) throw std::runtime_error("Failed to create command queue");
        
        const char* kernel_code = gelu_opencl_kernel;
        size_t kernel_length = strlen(kernel_code);
        ocl_state.prog = clCreateProgramWithSource(ocl_state.ctx, 1, &kernel_code, &kernel_length, &status);
        if (status != CL_SUCCESS) throw std::runtime_error("Failed to create program");
        
        status = clBuildProgram(ocl_state.prog, 1, &ocl_state.dev, nullptr, nullptr, nullptr);
        if (status != CL_SUCCESS) throw std::runtime_error("Failed to build program");
        
        ocl_state.kern = clCreateKernel(ocl_state.prog, "compute_gelu", &status);
        if (status != CL_SUCCESS) throw std::runtime_error("Failed to create kernel");
        
        ocl_state.ready = true;
    }
    
    size_t required_bytes = element_count * sizeof(float);
    
    if (ocl_state.buffer_capacity < required_bytes) {
        if (ocl_state.mem_in) clReleaseMemObject(ocl_state.mem_in);
        if (ocl_state.mem_out) clReleaseMemObject(ocl_state.mem_out);
        
        ocl_state.mem_in = clCreateBuffer(ocl_state.ctx, CL_MEM_READ_ONLY, required_bytes, nullptr, &status);
        ocl_state.mem_out = clCreateBuffer(ocl_state.ctx, CL_MEM_WRITE_ONLY, required_bytes, nullptr, &status);
        ocl_state.buffer_capacity = required_bytes;
    }
    
    clEnqueueWriteBuffer(ocl_state.cmd_queue, ocl_state.mem_in, CL_TRUE, 0, 
                        required_bytes, input.data(), 0, nullptr, nullptr);
    
    clSetKernelArg(ocl_state.kern, 0, sizeof(cl_mem), &ocl_state.mem_in);
    clSetKernelArg(ocl_state.kern, 1, sizeof(cl_mem), &ocl_state.mem_out);
    unsigned int elem_count_uint = static_cast<unsigned int>(element_count);
    clSetKernelArg(ocl_state.kern, 2, sizeof(unsigned int), &elem_count_uint);
    
    size_t work_group_size = 256;
    size_t global_work_size = ((element_count + work_group_size - 1) / work_group_size) * work_group_size;
    
    clEnqueueNDRangeKernel(ocl_state.cmd_queue, ocl_state.kern, 1, nullptr, 
                          &global_work_size, &work_group_size, 0, nullptr, nullptr);
    
    clEnqueueReadBuffer(ocl_state.cmd_queue, ocl_state.mem_out, CL_TRUE, 0, 
                       required_bytes, result.data(), 0, nullptr, nullptr);
    
    return result;
}
