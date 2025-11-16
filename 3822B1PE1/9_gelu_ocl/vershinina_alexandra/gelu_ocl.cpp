#include "gelu_ocl.h"

#include <CL/cl.h>
#include <vector>
#include <string>
#include <mutex>
#include <stdexcept>
#include <cstring>
#include <iostream>

namespace {
    const char kGeluKernelSrc[] = R"CLC(
__kernel void gelu_kernel(__global const float* input,
                          __global float* output,
                          const int n) {
    const int gid = get_global_id(0);
    if (gid >= n) return;

    float x = input[gid];
    // sqrt(2/pi)
    const float k = 0.7978845608028654f;
    const float c = 0.044715f;

    float x3 = x * x * x;
    float inner = k * (x + c * x3);

    // tanh(inner) via exp for performance: tanh(t) = (e^{2t} - 1)/(e^{2t} + 1)
    float e2t = exp(2.0f * inner);
    float tanh_val = (e2t - 1.0f) / (e2t + 1.0f);

    output[gid] = 0.5f * x * (1.0f + tanh_val);
}
)CLC";

    inline void ThrowClError(cl_int err, const char* msg) {
        if (err != CL_SUCCESS) {
            throw std::runtime_error(std::string(msg) + " (cl error " + std::to_string(err) + ")");
        }
    }

    struct OclContext {
        cl_platform_id platform = nullptr;
        cl_device_id device = nullptr;
        cl_context context = nullptr;
        cl_command_queue queue = nullptr;
        cl_program program = nullptr;
        cl_kernel kernel = nullptr;
        int platform_index = -1;
        bool initialized = false;
    };

    OclContext g_ctx;
    std::mutex g_ctx_mutex;

    void InitOpenCLOnce(int platform_index) {
        std::lock_guard<std::mutex> lock(g_ctx_mutex);
        if (g_ctx.initialized && g_ctx.platform_index == platform_index) return;
        if (g_ctx.initialized) {
            if (g_ctx.kernel) clReleaseKernel(g_ctx.kernel);
            if (g_ctx.program) clReleaseProgram(g_ctx.program);
            if (g_ctx.queue) clReleaseCommandQueue(g_ctx.queue);
            if (g_ctx.context) clReleaseContext(g_ctx.context);

            g_ctx = OclContext();
        }

        cl_int err = CL_SUCCESS;

        cl_uint num_platforms = 0;
        err = clGetPlatformIDs(0, nullptr, &num_platforms);
        ThrowClError(err, "clGetPlatformIDs (count)");
        if (num_platforms == 0) throw std::runtime_error("No OpenCL platforms found");

        std::vector<cl_platform_id> platforms(num_platforms);
        err = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
        ThrowClError(err, "clGetPlatformIDs (list)");

        if (platform_index < 0 || platform_index >= static_cast<int>(num_platforms)) {
            platform_index = 0;
        }
        cl_platform_id chosen_platform = platforms[platform_index];

        cl_uint num_devices = 0;
        err = clGetDeviceIDs(chosen_platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
        ThrowClError(err, "clGetDeviceIDs (count)");
        if (num_devices == 0) throw std::runtime_error("No GPU devices found on chosen platform");

        std::vector<cl_device_id> devices(num_devices);
        err = clGetDeviceIDs(chosen_platform, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr);
        ThrowClError(err, "clGetDeviceIDs (list)");

        cl_device_id device = devices[0]; 

        cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        ThrowClError(err, "clCreateContext");

#if defined(CL_VERSION_2_0)
        cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
#else
        cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
#endif
        ThrowClError(err, "clCreateCommandQueue");

        const char* src = kGeluKernelSrc;
        size_t src_len = std::strlen(src);
        cl_program program = clCreateProgramWithSource(context, 1, &src, &src_len, &err);
        ThrowClError(err, "clCreateProgramWithSource");

        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t log_size = 0;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::vector<char> log(log_size ? log_size : 1);
            if (log_size) clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            std::string log_s(log.begin(), log.end());

            clReleaseProgram(program);
            clReleaseCommandQueue(queue);
            clReleaseContext(context);
            throw std::runtime_error(std::string("clBuildProgram failed: ") + log_s);
        }

        cl_kernel kernel = clCreateKernel(program, "gelu_kernel", &err);
        ThrowClError(err, "clCreateKernel");

        g_ctx.platform = chosen_platform;
        g_ctx.device = device;
        g_ctx.context = context;
        g_ctx.queue = queue;
        g_ctx.program = program;
        g_ctx.kernel = kernel;
        g_ctx.platform_index = platform_index;
        g_ctx.initialized = true;
    }
} // namespace

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    if (input.empty()) return {};

    InitOpenCLOnce(platform);

    cl_int err = CL_SUCCESS;
    const size_t n = input.size();
    const size_t bytes = n * sizeof(float);
    cl_mem_flags in_flags = CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR;
    cl_mem_flags out_flags = CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR;

    cl_mem buf_in = clCreateBuffer(g_ctx.context, in_flags, bytes, nullptr, &err);
    ThrowClError(err, "clCreateBuffer (input)");

    cl_mem buf_out = clCreateBuffer(g_ctx.context, out_flags, bytes, nullptr, &err);
    ThrowClError(err, "clCreateBuffer (output)");

    void* mapped_in = clEnqueueMapBuffer(g_ctx.queue, buf_in, CL_TRUE, CL_MAP_WRITE, 0, bytes, 0, nullptr, nullptr, &err);
    ThrowClError(err, "clEnqueueMapBuffer (input)");
    std::memcpy(mapped_in, input.data(), bytes);
    err = clEnqueueUnmapMemObject(g_ctx.queue, buf_in, mapped_in, 0, nullptr, nullptr);
    ThrowClError(err, "clEnqueueUnmapMemObject (input)");

    err = clSetKernelArg(g_ctx.kernel, 0, sizeof(cl_mem), &buf_in);
    err |= clSetKernelArg(g_ctx.kernel, 1, sizeof(cl_mem), &buf_out);
    int n_int = static_cast<int>(n);
    err |= clSetKernelArg(g_ctx.kernel, 2, sizeof(int), &n_int);
    ThrowClError(err, "clSetKernelArg");

    const size_t local_size = 256;
    const size_t global_size = ((n + local_size - 1) / local_size) * local_size;

    err = clEnqueueNDRangeKernel(g_ctx.queue, g_ctx.kernel, 1, nullptr, &global_size, &local_size, 0, nullptr, nullptr);
    ThrowClError(err, "clEnqueueNDRangeKernel");

    void* mapped_out = clEnqueueMapBuffer(g_ctx.queue, buf_out, CL_TRUE, CL_MAP_READ, 0, bytes, 0, nullptr, nullptr, &err);
    ThrowClError(err, "clEnqueueMapBuffer (output)");

    std::vector<float> result(n);
    std::memcpy(result.data(), mapped_out, bytes);

    err = clEnqueueUnmapMemObject(g_ctx.queue, buf_out, mapped_out, 0, nullptr, nullptr);
    ThrowClError(err, "clEnqueueUnmapMemObject (output)");

    err = clFinish(g_ctx.queue);
    ThrowClError(err, "clFinish");

    clReleaseMemObject(buf_in);
    clReleaseMemObject(buf_out);

    return result;
}
