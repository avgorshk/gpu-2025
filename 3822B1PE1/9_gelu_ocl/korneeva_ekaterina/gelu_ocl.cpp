#define CL_TARGET_OPENCL_VERSION 300

#include "gelu_ocl.h"
#include <CL/cl.h>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

static const char* gelu_opencl_kernel = R"(
__kernel void compute_gelu(__global const float* restrict in_data, 
                           __global float* restrict out_data, 
                           const unsigned int count) {
    const unsigned int gid = get_global_id(0);
    
    for(unsigned int i = 0; i < 4; i++) {
        unsigned int idx = gid * 4 + i;
        if (idx >= count) return;
        
        float val = in_data[idx];
        float val_cubed = val * val * val;

        float inner = 0.7978845608f * (val + 0.044715f * val_cubed);
        
        float exp_val = exp(2.0f * inner);
        float tanh_result = 1.0f - 2.0f / (exp_val + 1.0f);
        
        out_data[idx] = 0.5f * val * (1.0f + tanh_result);
    }
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
    size_t max_work_group_size;
    bool ready;
} ocl_state = { nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 256, false };

static void cleanup_opencl() {
    if (ocl_state.mem_in) clReleaseMemObject(ocl_state.mem_in);
    if (ocl_state.mem_out) clReleaseMemObject(ocl_state.mem_out);
    if (ocl_state.kern) clReleaseKernel(ocl_state.kern);
    if (ocl_state.prog) clReleaseProgram(ocl_state.prog);
    if (ocl_state.cmd_queue) clReleaseCommandQueue(ocl_state.cmd_queue);
    if (ocl_state.ctx) clReleaseContext(ocl_state.ctx);

    ocl_state = { nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 256, false };
}

static struct OpenCLCleanup {
    ~OpenCLCleanup() { cleanup_opencl(); }
} opencl_cleanup;

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    if (input.empty()) return {};

    const size_t element_count = input.size();
    std::vector<float> result(element_count);
    cl_int status;

    if (!ocl_state.ready) {
        cl_uint platform_count = 0;
        clGetPlatformIDs(0, nullptr, &platform_count);
        if (platform_count == 0) {
            throw std::runtime_error("No OpenCL platforms found");
        }

        std::vector<cl_platform_id> platforms(platform_count);
        clGetPlatformIDs(platform_count, platforms.data(), nullptr);

        int selected_platform = std::max(0, std::min(platform, static_cast<int>(platform_count - 1)));
        ocl_state.plat = platforms[selected_platform];

        cl_uint device_count = 0;
        status = clGetDeviceIDs(ocl_state.plat, CL_DEVICE_TYPE_GPU, 0, nullptr, &device_count);

        if (status != CL_SUCCESS || device_count == 0) {
            status = clGetDeviceIDs(ocl_state.plat, CL_DEVICE_TYPE_CPU, 0, nullptr, &device_count);
            if (status != CL_SUCCESS || device_count == 0) {
                throw std::runtime_error("No OpenCL devices found (GPU or CPU)");
            }
        }

        std::vector<cl_device_id> devices(device_count);
        cl_device_type device_type = (device_count > 0 && status == CL_SUCCESS) ?
            CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU;
        clGetDeviceIDs(ocl_state.plat, device_type, device_count, devices.data(), nullptr);

        ocl_state.dev = devices[0];

        clGetDeviceInfo(ocl_state.dev, CL_DEVICE_MAX_WORK_GROUP_SIZE,
            sizeof(size_t), &ocl_state.max_work_group_size, nullptr);

        ocl_state.ctx = clCreateContext(nullptr, 1, &ocl_state.dev, nullptr, nullptr, &status);
        if (status != CL_SUCCESS) {
            cleanup_opencl();
            throw std::runtime_error("Failed to create OpenCL context");
        }

        ocl_state.cmd_queue = clCreateCommandQueueWithProperties(ocl_state.ctx, ocl_state.dev,
            0, &status);
        if (status != CL_SUCCESS) {
            cleanup_opencl();
            throw std::runtime_error("Failed to create command queue");
        }

        const char* kernel_code = gelu_opencl_kernel;
        ocl_state.prog = clCreateProgramWithSource(ocl_state.ctx, 1, &kernel_code,
            nullptr, &status);
        if (status != CL_SUCCESS) {
            cleanup_opencl();
            throw std::runtime_error("Failed to create program");
        }

        status = clBuildProgram(ocl_state.prog, 1, &ocl_state.dev,
            "-cl-fast-relaxed-math -cl-mad-enable", nullptr, nullptr);
        if (status != CL_SUCCESS) {
            size_t log_size;
            clGetProgramBuildInfo(ocl_state.prog, ocl_state.dev, CL_PROGRAM_BUILD_LOG,
                0, nullptr, &log_size);
            std::vector<char> log(log_size);
            clGetProgramBuildInfo(ocl_state.prog, ocl_state.dev, CL_PROGRAM_BUILD_LOG,
                log_size, log.data(), nullptr);

            cleanup_opencl();
            throw std::runtime_error("Failed to build program: " + std::string(log.data()));
        }

        ocl_state.kern = clCreateKernel(ocl_state.prog, "compute_gelu", &status);
        if (status != CL_SUCCESS) {
            cleanup_opencl();
            throw std::runtime_error("Failed to create kernel");
        }

        ocl_state.ready = true;
    }

    const size_t required_bytes = element_count * sizeof(float);

    if (ocl_state.buffer_capacity < required_bytes) {
        if (ocl_state.mem_in) clReleaseMemObject(ocl_state.mem_in);
        if (ocl_state.mem_out) clReleaseMemObject(ocl_state.mem_out);

        ocl_state.mem_in = clCreateBuffer(ocl_state.ctx, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
            required_bytes, nullptr, &status);
        if (status != CL_SUCCESS) {
            cleanup_opencl();
            throw std::runtime_error("Failed to create input buffer");
        }

        ocl_state.mem_out = clCreateBuffer(ocl_state.ctx, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
            required_bytes, nullptr, &status);
        if (status != CL_SUCCESS) {
            cleanup_opencl();
            throw std::runtime_error("Failed to create output buffer");
        }

        ocl_state.buffer_capacity = required_bytes;
    }

    cl_event write_event;
    status = clEnqueueWriteBuffer(ocl_state.cmd_queue, ocl_state.mem_in, CL_FALSE, 0,
        required_bytes, input.data(), 0, nullptr, &write_event);
    if (status != CL_SUCCESS) {
        cleanup_opencl();
        throw std::runtime_error("Failed to write input data");
    }

    status = clSetKernelArg(ocl_state.kern, 0, sizeof(cl_mem), &ocl_state.mem_in);
    status |= clSetKernelArg(ocl_state.kern, 1, sizeof(cl_mem), &ocl_state.mem_out);
    const unsigned int elem_count_uint = static_cast<unsigned int>(element_count);
    status |= clSetKernelArg(ocl_state.kern, 2, sizeof(unsigned int), &elem_count_uint);

    if (status != CL_SUCCESS) {
        cleanup_opencl();
        throw std::runtime_error("Failed to set kernel arguments");
    }

    size_t work_group_size = std::min(ocl_state.max_work_group_size, static_cast<size_t>(256));

    const size_t items_needed = (element_count + 3) / 4;
    size_t global_work_size = ((items_needed + work_group_size - 1) / work_group_size) * work_group_size;

    cl_event kernel_event;
    status = clEnqueueNDRangeKernel(ocl_state.cmd_queue, ocl_state.kern, 1, nullptr,
        &global_work_size, &work_group_size,
        1, &write_event, &kernel_event);
    if (status != CL_SUCCESS) {
        cleanup_opencl();
        throw std::runtime_error("Failed to execute kernel");
    }

    cl_event read_event;
    status = clEnqueueReadBuffer(ocl_state.cmd_queue, ocl_state.mem_out, CL_FALSE, 0,
        required_bytes, result.data(),
        1, &kernel_event, &read_event);
    if (status != CL_SUCCESS) {
        cleanup_opencl();
        throw std::runtime_error("Failed to read results");
    }

    clWaitForEvents(1, &read_event);

    clReleaseEvent(write_event);
    clReleaseEvent(kernel_event);
    clReleaseEvent(read_event);

    return result;
}