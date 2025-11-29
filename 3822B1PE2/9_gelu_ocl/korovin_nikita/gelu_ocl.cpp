#include "gelu_ocl.h"
#include <CL/opencl.h>
#include <mutex>

static cl_context g_context = nullptr;
static cl_command_queue g_queue = nullptr;
static cl_kernel g_kernel = nullptr;
static cl_program g_program = nullptr;
static bool g_initialized = false;
static std::mutex g_init_mutex;
static int g_cached_platform = -1;

const char* geluKernel = R"(
__kernel void compute_gelu(__global const float* input, __global float* output, int count) {
    int i = get_global_id(0);
    if (i < count) {
        float x = input[i];
        const float a = 0.7978845608028654f;
        const float b = 0.044715f;
        float x3 = x * x * x;
        float inner = a * (x + b * x3);
        float exp_pos = exp(inner);
        float exp_neg = exp(-inner);
        float tanh_val = (exp_pos - exp_neg) / (exp_pos + exp_neg);
        output[i] = 0.5f * x * (1.0f + tanh_val);
    }
}
)";

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    int size = input.size();
    if (size == 0) {
        return std::vector<float>();
    }

    std::vector<float> result(size);
    std::lock_guard<std::mutex> lock(g_init_mutex);
    
    if (!g_initialized || g_cached_platform != platform) {
        if (g_initialized) {
            clReleaseKernel(g_kernel);
            clReleaseProgram(g_program);
            clReleaseCommandQueue(g_queue);
            clReleaseContext(g_context);
        }
        cl_uint numPlatforms;
        clGetPlatformIDs(0, NULL, &numPlatforms);
        std::vector<cl_platform_id> platforms(numPlatforms);
        clGetPlatformIDs(numPlatforms, platforms.data(), NULL);
        
        cl_device_id device;
        clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_GPU, 1, &device, NULL);

        g_context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
        g_queue = clCreateCommandQueue(g_context, device, 0, NULL);
        g_program = clCreateProgramWithSource(g_context, 1, &geluKernel, NULL, NULL);
        clBuildProgram(g_program, 1, &device, NULL, NULL, NULL);
        g_kernel = clCreateKernel(g_program, "compute_gelu", NULL);
        g_initialized = true;
        g_cached_platform = platform;
    }

    size_t bytes = size * sizeof(float);
    cl_mem inputBuffer = clCreateBuffer(g_context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    cl_mem outputBuffer = clCreateBuffer(g_context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);
    
    cl_event writeEvent;
    clEnqueueWriteBuffer(g_queue, inputBuffer, CL_FALSE, 0, bytes, input.data(), 0, NULL, &writeEvent);
    clSetKernelArg(g_kernel, 0, sizeof(cl_mem), &inputBuffer);
    clSetKernelArg(g_kernel, 1, sizeof(cl_mem), &outputBuffer);
    clSetKernelArg(g_kernel, 2, sizeof(int), &size);
    
    size_t globalSize = (size + 255) / 256 * 256;
    size_t localSize = 256;
    cl_event kernelEvent;
    clEnqueueNDRangeKernel(g_queue, g_kernel, 1, NULL, &globalSize, &localSize, 1, &writeEvent, &kernelEvent);
    
    cl_event readEvent;
    clEnqueueReadBuffer(g_queue, outputBuffer, CL_FALSE, 0, bytes, result.data(), 1, &kernelEvent, &readEvent);
    clWaitForEvents(1, &readEvent);
    clReleaseEvent(readEvent);
    clReleaseEvent(kernelEvent);
    clReleaseEvent(writeEvent);
    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);
    
    return result;
}